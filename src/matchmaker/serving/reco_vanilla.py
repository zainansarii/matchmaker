"""
FAISS-based recommender that operates purely on ALS factors without league constraints.

This module mirrors the league-aware recommender but indexes all candidates of a given
gender together, enabling ablation studies that compare performance with and without
league filtering.
"""

from __future__ import annotations

import numpy as np
import faiss
from typing import Dict, List, Tuple, Any, Optional


class ALSFaissRecommender:
    """FAISS-backed recommender that uses ALS factors without league filtering."""

    def __init__(self, als_model, use_gpu: bool = True):
        self.als_model = als_model
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0

        self.female_index: Optional[Any] = None
        self.female_id_map: List[int] = []
        self.male_index: Optional[Any] = None
        self.male_id_map: List[int] = []

        self._build_gender_indices()

    def _build_gender_indices(self) -> None:
        """Build FAISS indices for female and male candidate pools."""

        # Females as candidates for males
        female_ids = sorted(set(self.als_model.female_map.keys()) | set(self.als_model.female_map_f2m.keys()))
        if female_ids:
            female_vectors = []
            valid_female_ids = []
            for fid in female_ids:
                if fid in self.als_model.female_map:
                    idx = self.als_model.female_map[fid]
                    female_vectors.append(self.als_model.female_factors[idx].get())
                    valid_female_ids.append(fid)
                elif fid in self.als_model.female_map_f2m:
                    idx = self.als_model.female_map_f2m[fid]
                    female_vectors.append(self.als_model.female_pref_factors[idx].get())
                    valid_female_ids.append(fid)

            if female_vectors:
                factors = np.vstack(female_vectors)
                self.female_index = self._create_index(factors)
                self.female_id_map = valid_female_ids

        # Males as candidates for females
        male_ids = sorted(set(self.als_model.male_map.keys()) | set(self.als_model.male_map_f2m.keys()))
        if male_ids:
            male_vectors = []
            valid_male_ids = []
            for mid in male_ids:
                if mid in self.als_model.male_map:
                    idx = self.als_model.male_map[mid]
                    male_vectors.append(self.als_model.male_factors[idx].get())
                    valid_male_ids.append(mid)
                elif mid in self.als_model.male_map_f2m:
                    idx = self.als_model.male_map_f2m[mid]
                    male_vectors.append(self.als_model.male_attr_factors[idx].get())
                    valid_male_ids.append(mid)

            if male_vectors:
                factors = np.vstack(male_vectors)
                self.male_index = self._create_index(factors)
                self.male_id_map = valid_male_ids

    def _create_index(self, factors: np.ndarray) -> Any:
        """Create a FAISS index from factor embeddings."""
        d = factors.shape[1]
        n = factors.shape[0]

        use_gpu_for_index = self.use_gpu and n > 5000

        if n < 100000:
            index = faiss.IndexFlatIP(d)
        else:
            nlist = min(int(np.sqrt(n)), 4096)
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(factors.astype("float32"))
            index.nprobe = min(32, nlist)

        if use_gpu_for_index:
            try:
                res = faiss.StandardGpuResources()
                res.setTempMemory(256 * 1024 * 1024)
                index = faiss.index_cpu_to_gpu(res, 0, index)
            except Exception as exc:  # pragma: no cover - defensive, GPU specific
                print(f"Warning: Falling back to CPU FAISS index ({n} vectors): {exc}")

        index.add(factors.astype("float32"))
        return index

    # ------------------------------------------------------------------
    # Individual recommendation entry-points
    # ------------------------------------------------------------------

    def recommend_for_male(self, male_id: int, k: int = 1000) -> List[Tuple[int, float]]:
        """Return top-k female candidates for a male user."""
        if self.female_index is None:
            return []
        if male_id not in self.als_model.male_map:
            return []

        m_idx = self.als_model.male_map[male_id]
        male_factor = self.als_model.male_factors[m_idx].get().reshape(1, -1)

        k_search = min(k, len(self.female_id_map))
        distances, indices = self.female_index.search(male_factor.astype("float32"), k_search)

        candidate_ids = []
        for idx in indices[0]:
            if idx == -1:
                continue
            candidate_ids.append(self.female_id_map[idx])

        if not candidate_ids:
            return []

        pairs = [(male_id, fid) for fid in candidate_ids]
        scores = self.als_model.mutual_score_batch(pairs)

        recs = [(fid, float(score)) for fid, score in zip(candidate_ids, scores) if score > 0]
        recs.sort(key=lambda x: x[1], reverse=True)
        return recs[:k]

    def recommend_for_female(self, female_id: int, k: int = 1000) -> List[Tuple[int, float]]:
        """Return top-k male candidates for a female user."""
        if self.male_index is None:
            return []
        if female_id not in self.als_model.female_map_f2m:
            return []

        f_idx = self.als_model.female_map_f2m[female_id]
        female_factor = self.als_model.female_pref_factors[f_idx].get().reshape(1, -1)

        k_search = min(k, len(self.male_id_map))
        distances, indices = self.male_index.search(female_factor.astype("float32"), k_search)

        candidate_ids = []
        for idx in indices[0]:
            if idx == -1:
                continue
            candidate_ids.append(self.male_id_map[idx])

        if not candidate_ids:
            return []

        pairs = [(mid, female_id) for mid in candidate_ids]
        scores = self.als_model.mutual_score_batch(pairs)

        recs = [(mid, float(score)) for mid, score in zip(candidate_ids, scores) if score > 0]
        recs.sort(key=lambda x: x[1], reverse=True)
        return recs[:k]

    # ------------------------------------------------------------------
    # Batch interfaces
    # ------------------------------------------------------------------

    def recommend_batch(self, user_ids: List[int], user_metadata: Dict[int, Dict[str, str]],
                        k: int = 1000) -> Dict[int, List[Tuple[int, float]]]:
        """Batch recommend for multiple users grouped only by gender."""
        groups: Dict[str, List[int]] = {"M": [], "F": []}
        results: Dict[int, List[Tuple[int, float]]] = {}

        for uid in user_ids:
            meta = user_metadata.get(uid)
            if not meta:
                results[uid] = []
                continue
            gender = meta.get("gender")
            if gender in groups:
                groups[gender].append(uid)
            else:
                results[uid] = []

        if groups["M"]:
            results.update(self._batch_recommend_males(groups["M"], k))
        if groups["F"]:
            results.update(self._batch_recommend_females(groups["F"], k))

        # Ensure any users with missing metadata are present
        for uid in user_ids:
            results.setdefault(uid, [])

        return results

    def _batch_recommend_males(self, male_ids: List[int], k: int) -> Dict[int, List[Tuple[int, float]]]:
        if self.female_index is None:
            return {mid: [] for mid in male_ids}

        valid_pairs = []
        for mid in male_ids:
            if mid in self.als_model.male_map:
                idx = self.als_model.male_map[mid]
                valid_pairs.append((mid, self.als_model.male_factors[idx].get()))

        if not valid_pairs:
            return {mid: [] for mid in male_ids}

        male_ids_ordered, factors = zip(*valid_pairs)
        query_matrix = np.vstack(factors).astype("float32")

        k_search = min(k, len(self.female_id_map))
        distances, indices = self.female_index.search(query_matrix, k_search)

        all_pairs: List[Tuple[int, int]] = []
        pairs_by_user: Dict[int, List[int]] = {mid: [] for mid in male_ids_ordered}
        for row_idx, mid in enumerate(male_ids_ordered):
            for col_idx in indices[row_idx]:
                if col_idx == -1:
                    continue
                fid = self.female_id_map[col_idx]
                all_pairs.append((mid, fid))
                pairs_by_user[mid].append(fid)

        if not all_pairs:
            return {mid: [] for mid in male_ids}

        scores = self.als_model.mutual_score_batch(all_pairs)

        results: Dict[int, List[Tuple[int, float]]] = {mid: [] for mid in male_ids}
        score_iter = iter(scores)
        for mid in male_ids_ordered:
            recs = []
            for fid in pairs_by_user[mid]:
                score = float(next(score_iter))
                if score > 0:
                    recs.append((fid, score))
            recs.sort(key=lambda x: x[1], reverse=True)
            results[mid] = recs[:k]

        return results

    def _batch_recommend_females(self, female_ids: List[int], k: int) -> Dict[int, List[Tuple[int, float]]]:
        if self.male_index is None:
            return {fid: [] for fid in female_ids}

        valid_pairs = []
        for fid in female_ids:
            if fid in self.als_model.female_map_f2m:
                idx = self.als_model.female_map_f2m[fid]
                valid_pairs.append((fid, self.als_model.female_pref_factors[idx].get()))

        if not valid_pairs:
            return {fid: [] for fid in female_ids}

        female_ids_ordered, factors = zip(*valid_pairs)
        query_matrix = np.vstack(factors).astype("float32")

        k_search = min(k, len(self.male_id_map))
        distances, indices = self.male_index.search(query_matrix, k_search)

        all_pairs: List[Tuple[int, int]] = []
        pairs_by_user: Dict[int, List[int]] = {fid: [] for fid in female_ids_ordered}
        for row_idx, fid in enumerate(female_ids_ordered):
            for col_idx in indices[row_idx]:
                if col_idx == -1:
                    continue
                mid = self.male_id_map[col_idx]
                all_pairs.append((mid, fid))
                pairs_by_user[fid].append(mid)

        if not all_pairs:
            return {fid: [] for fid in female_ids}

        scores = self.als_model.mutual_score_batch(all_pairs)

        results: Dict[int, List[Tuple[int, float]]] = {fid: [] for fid in female_ids}
        score_iter = iter(scores)
        for fid in female_ids_ordered:
            recs = []
            for mid in pairs_by_user[fid]:
                score = float(next(score_iter))
                if score > 0:
                    recs.append((mid, score))
            recs.sort(key=lambda x: x[1], reverse=True)
            results[fid] = recs[:k]

        return results
