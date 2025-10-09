"""GPU-only FAISS recommender that uses ALS factors without league filtering."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import faiss  # type: ignore
import numpy as np


class ALSFaissRecommender:
    """Simplified GPU-only recommender without league constraints."""

    def __init__(self, als_model, use_gpu: bool = True):
        if not use_gpu:
            raise ValueError("CPU execution is no longer supported; set use_gpu=True.")
        if faiss.get_num_gpus() == 0:
            raise RuntimeError(
                "ALSFaissRecommender requires faiss-gpu with at least one CUDA device."
            )

        self.als_model = als_model
        self.gpu_resources = faiss.StandardGpuResources()
        self.gpu_resources.setTempMemory(256 * 1024 * 1024)

        self.female_index, self.female_id_map = self._build_candidate_index(
            sorted(
                set(self.als_model.female_map.keys())
                | set(self.als_model.female_map_f2m.keys())
            ),
            primary_map=self.als_model.female_map,
            primary_factors=self.als_model.female_factors,
            fallback_map=self.als_model.female_map_f2m,
            fallback_factors=self.als_model.female_pref_factors,
        )

        self.male_index, self.male_id_map = self._build_candidate_index(
            sorted(
                set(self.als_model.male_map.keys()) | set(self.als_model.male_map_f2m.keys())
            ),
            primary_map=self.als_model.male_map,
            primary_factors=self.als_model.male_factors,
            fallback_map=self.als_model.male_map_f2m,
            fallback_factors=self.als_model.male_attr_factors,
        )

    def recommend_for_male(self, male_id: int, k: int = 1000) -> List[Tuple[int, float]]:
        """Return top-k female candidates for a male user."""

        return self._recommend_single(
            user_id=male_id,
            query_map=self.als_model.male_map,
            query_factors=self.als_model.male_factors,
            candidate_index=self.female_index,
            candidate_ids=self.female_id_map,
            pair_builder=lambda uid, cid: (uid, cid),
            k=k,
        )

    def recommend_for_female(self, female_id: int, k: int = 1000) -> List[Tuple[int, float]]:
        """Return top-k male candidates for a female user."""

        return self._recommend_single(
            user_id=female_id,
            query_map=self.als_model.female_map_f2m,
            query_factors=self.als_model.female_pref_factors,
            candidate_index=self.male_index,
            candidate_ids=self.male_id_map,
            pair_builder=lambda uid, cid: (cid, uid),
            k=k,
        )

    def recommend_batch(
        self,
        user_ids: List[int],
        user_metadata: Dict[int, Dict[str, str]],
        k: int = 1000,
    ) -> Dict[int, List[Tuple[int, float]]]:
        """Batch recommend for multiple users grouped by gender."""

        results: Dict[int, List[Tuple[int, float]]] = {}
        male_ids: List[int] = []
        female_ids: List[int] = []

        for uid in user_ids:
            meta = user_metadata.get(uid)
            if not meta:
                results[uid] = []
                continue

            gender = meta.get("gender")
            if gender == "M":
                male_ids.append(uid)
            elif gender == "F":
                female_ids.append(uid)
            else:
                results[uid] = []

        if male_ids:
            results.update(
                self._batch_recommend(
                    user_ids=male_ids,
                    query_map=self.als_model.male_map,
                    query_factors=self.als_model.male_factors,
                    candidate_index=self.female_index,
                    candidate_ids=self.female_id_map,
                    pair_builder=lambda uid, cid: (uid, cid),
                    k=k,
                )
            )

        if female_ids:
            results.update(
                self._batch_recommend(
                    user_ids=female_ids,
                    query_map=self.als_model.female_map_f2m,
                    query_factors=self.als_model.female_pref_factors,
                    candidate_index=self.male_index,
                    candidate_ids=self.male_id_map,
                    pair_builder=lambda uid, cid: (cid, uid),
                    k=k,
                )
            )

        for uid in user_ids:
            results.setdefault(uid, [])

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_candidate_index(
        self,
        user_ids: List[int],
        primary_map: Dict[int, int],
        primary_factors: Any,
        fallback_map: Dict[int, int],
        fallback_factors: Any,
    ) -> Tuple[Optional[Any], List[int]]:
        vectors, valid_ids = self._collect_candidate_vectors(
            user_ids, primary_map, primary_factors, fallback_map, fallback_factors
        )

        if vectors is None:
            return None, []

        return self._create_gpu_index(vectors), valid_ids

    def _collect_candidate_vectors(
        self,
        user_ids: List[int],
        primary_map: Dict[int, int],
        primary_factors: Any,
        fallback_map: Dict[int, int],
        fallback_factors: Any,
    ) -> Tuple[Optional[np.ndarray], List[int]]:
        vectors: List[np.ndarray] = []
        valid_ids: List[int] = []

        for uid in user_ids:
            if uid in primary_map:
                idx = primary_map[uid]
                vectors.append(np.asarray(primary_factors[idx].get(), dtype=np.float32))
                valid_ids.append(uid)
            elif uid in fallback_map:
                idx = fallback_map[uid]
                vectors.append(np.asarray(fallback_factors[idx].get(), dtype=np.float32))
                valid_ids.append(uid)

        if not vectors:
            return None, []

        stacked = np.vstack(vectors).astype("float32")
        return stacked, valid_ids

    def _create_gpu_index(self, vectors: np.ndarray) -> Any:
        dim = vectors.shape[1]
        index = faiss.GpuIndexFlatIP(self.gpu_resources, dim)
        index.add(vectors.astype("float32", copy=False))
        return index

    def _extract_query_vector(
        self, user_id: int, query_map: Dict[int, int], query_factors: Any
    ) -> Optional[np.ndarray]:
        idx = query_map.get(user_id)
        if idx is None:
            return None

        vector = np.asarray(query_factors[idx].get(), dtype=np.float32)
        return vector.reshape(1, -1)

    def _recommend_single(
        self,
        user_id: int,
        query_map: Dict[int, int],
        query_factors: Any,
        candidate_index: Optional[Any],
        candidate_ids: List[int],
        pair_builder,
        k: int,
    ) -> List[Tuple[int, float]]:
        if candidate_index is None or not candidate_ids:
            return []

        query_vector = self._extract_query_vector(user_id, query_map, query_factors)
        if query_vector is None:
            return []

        candidate_list = self._search_candidates(query_vector, candidate_index, candidate_ids, k)
        if not candidate_list:
            return []

        pairs = [pair_builder(user_id, cid) for cid in candidate_list]
        scores = np.asarray(self.als_model.mutual_score_batch(pairs), dtype=np.float32)

        recommendations = [
            (cid, float(score)) for cid, score in zip(candidate_list, scores) if score > 0
        ]
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:k]

    def _collect_query_matrix(
        self,
        user_ids: List[int],
        query_map: Dict[int, int],
        query_factors: Any,
    ) -> Tuple[List[int], Optional[np.ndarray]]:
        vectors: List[np.ndarray] = []
        valid_ids: List[int] = []

        for uid in user_ids:
            idx = query_map.get(uid)
            if idx is None:
                continue

            vectors.append(np.asarray(query_factors[idx].get(), dtype=np.float32))
            valid_ids.append(uid)

        if not vectors:
            return [], None

        matrix = np.vstack(vectors).astype("float32")
        return valid_ids, matrix

    def _batch_recommend(
        self,
        user_ids: List[int],
        query_map: Dict[int, int],
        query_factors: Any,
        candidate_index: Optional[Any],
        candidate_ids: List[int],
        pair_builder,
        k: int,
    ) -> Dict[int, List[Tuple[int, float]]]:
        results = {uid: [] for uid in user_ids}

        if candidate_index is None or not candidate_ids:
            return results

        valid_ids, query_matrix = self._collect_query_matrix(user_ids, query_map, query_factors)
        if query_matrix is None:
            return results

        k_search = min(k, len(candidate_ids))
        _, candidate_matrix = candidate_index.search(query_matrix, k_search)

        pairs: List[Tuple[int, int]] = []
        owners: List[int] = []
        scored_candidates: List[int] = []

        for row_idx, uid in enumerate(valid_ids):
            for candidate_idx in candidate_matrix[row_idx]:
                if candidate_idx == -1:
                    continue
                candidate_id = candidate_ids[candidate_idx]
                pairs.append(pair_builder(uid, candidate_id))
                owners.append(uid)
                scored_candidates.append(candidate_id)

        if not pairs:
            return results

        scores = np.asarray(self.als_model.mutual_score_batch(pairs), dtype=np.float32)
        per_user: Dict[int, List[Tuple[int, float]]] = {}

        for owner, candidate_id, score in zip(owners, scored_candidates, scores):
            if score > 0:
                per_user.setdefault(owner, []).append((candidate_id, float(score)))

        for uid in user_ids:
            recs = per_user.get(uid, [])
            recs.sort(key=lambda x: x[1], reverse=True)
            results[uid] = recs[:k]

        return results

    def _search_candidates(
        self, query: np.ndarray, index: Any, id_map: List[int], limit: int
    ) -> List[int]:
        if not id_map:
            return []

        k_search = min(limit, len(id_map))
        _, candidate_indices = index.search(query.astype("float32", copy=False), k_search)
        return [id_map[idx] for idx in candidate_indices[0] if idx != -1]
