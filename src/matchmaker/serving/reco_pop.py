"""GPU-only FAISS recommender for league-aware matching."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import faiss  # type: ignore
import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency for user convenience
    import cudf  # type: ignore
except ImportError:  # pragma: no cover - cudf not required for CPU tests
    cudf = None


class LeagueFilteredRecommender:
    """GPU-only FAISS recommender that filters candidates by league before scoring."""

    LEAGUES: Tuple[str, ...] = ("Bronze", "Silver", "Gold", "Platinum", "Diamond")
    OVERFETCH_MULTIPLIER: float = 3.0

    def __init__(self, als_model, user_df: pd.DataFrame, use_gpu: bool = True):
        if not use_gpu:
            raise ValueError("CPU execution is no longer supported; set use_gpu=True.")
        if faiss.get_num_gpus() == 0:
            raise RuntimeError(
                "LeagueFilteredRecommender requires faiss-gpu with at least one CUDA device."
            )

        self.als_model = als_model
        self.gpu_resources = faiss.StandardGpuResources()
        self.gpu_resources.setTempMemory(256 * 1024 * 1024)

        if cudf is not None and isinstance(user_df, cudf.DataFrame):
            user_df = user_df.to_pandas()
        elif not isinstance(user_df, pd.DataFrame):
            raise TypeError("user_df must be a pandas or cudf DataFrame.")

        missing_cols = {"user_id", "gender", "league"} - set(user_df.columns)
        if missing_cols:
            raise ValueError(f"user_df is missing required columns: {sorted(missing_cols)}")

        males_df = user_df[user_df["gender"] == "M"]
        females_df = user_df[user_df["gender"] == "F"]

        self.male_indices, self.male_id_map = self._build_indices_by_league(
            frame=males_df,
            primary_map=self.als_model.male_map,
            primary_factors=self.als_model.male_factors,
            fallback_map=self.als_model.male_map_f2m,
            fallback_factors=self.als_model.male_attr_factors,
        )
        self.female_indices, self.female_id_map = self._build_indices_by_league(
            frame=females_df,
            primary_map=self.als_model.female_map,
            primary_factors=self.als_model.female_factors,
            fallback_map=self.als_model.female_map_f2m,
            fallback_factors=self.als_model.female_pref_factors,
        )

    def recommend_for_male(self, male_id: int, league: str, k: int = 1000) -> List[Tuple[int, float]]:
        """Return top-k female candidates for a male user in the same league."""

        return self._recommend_single(
            user_id=male_id,
            league=league,
            query_map=self.als_model.male_map,
            query_factors=self.als_model.male_factors,
            candidate_indices=self.female_indices,
            candidate_id_map=self.female_id_map,
            pair_builder=lambda uid, cid: (uid, cid),
            k=k,
        )

    def recommend_for_female(self, female_id: int, league: str, k: int = 1000) -> List[Tuple[int, float]]:
        """Return top-k male candidates for a female user in the same league."""

        return self._recommend_single(
            user_id=female_id,
            league=league,
            query_map=self.als_model.female_map_f2m,
            query_factors=self.als_model.female_pref_factors,
            candidate_indices=self.male_indices,
            candidate_id_map=self.male_id_map,
            pair_builder=lambda uid, cid: (cid, uid),
            k=k,
        )

    def recommend_batch(
        self,
        user_ids: List[int],
        user_metadata: Dict[int, Dict[str, str]],
        k: int = 1000,
    ) -> Dict[int, List[Tuple[int, float]]]:
        """Batch recommend for multiple users grouped by gender and league."""

        results: Dict[int, List[Tuple[int, float]]] = {}
        groups: Dict[Tuple[str, str], List[int]] = {}

        for uid in user_ids:
            meta = user_metadata.get(uid)
            if not meta:
                results[uid] = []
                continue

            gender = meta.get("gender")
            league = meta.get("league")
            if gender not in {"M", "F"} or not league:
                results[uid] = []
                continue

            groups.setdefault((gender, league), []).append(uid)

        for (gender, league), group_ids in groups.items():
            if gender == "M":
                group_results = self._batch_recommend(
                    user_ids=group_ids,
                    league=league,
                    query_map=self.als_model.male_map,
                    query_factors=self.als_model.male_factors,
                    candidate_indices=self.female_indices,
                    candidate_id_map=self.female_id_map,
                    pair_builder=lambda uid, cid: (uid, cid),
                    k=k,
                )
            else:
                group_results = self._batch_recommend(
                    user_ids=group_ids,
                    league=league,
                    query_map=self.als_model.female_map_f2m,
                    query_factors=self.als_model.female_pref_factors,
                    candidate_indices=self.male_indices,
                    candidate_id_map=self.male_id_map,
                    pair_builder=lambda uid, cid: (cid, uid),
                    k=k,
                )

            results.update(group_results)

        for uid in user_ids:
            results.setdefault(uid, [])

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _allowed_leagues(self, league: str) -> Tuple[str, ...]:
        if league not in self.LEAGUES:
            return (league,)
        idx = self.LEAGUES.index(league)
        start = max(0, idx - 1)
        end = min(len(self.LEAGUES), idx + 2)
        return self.LEAGUES[start:end]

    @staticmethod
    def _dedupe_preserve_order(items: List[int]) -> List[int]:
        seen: set[int] = set()
        deduped: List[int] = []
        for item in items:
            if item not in seen:
                seen.add(item)
                deduped.append(item)
        return deduped

    def _build_indices_by_league(
        self,
        frame: pd.DataFrame,
        primary_map: Dict[int, int],
        primary_factors: Any,
        fallback_map: Dict[int, int],
        fallback_factors: Any,
    ) -> Tuple[Dict[str, Any], Dict[str, List[int]]]:
        indices: Dict[str, Any] = {}
        id_maps: Dict[str, List[int]] = {}

        for league in self.LEAGUES:
            league_ids = frame.loc[frame["league"] == league, "user_id"].tolist()
            if not league_ids:
                continue

            vectors, valid_ids = self._collect_candidate_vectors(
                league_ids,
                primary_map,
                primary_factors,
                fallback_map,
                fallback_factors,
            )

            if vectors is None:
                continue

            indices[league] = self._create_gpu_index(vectors)
            id_maps[league] = valid_ids

        return indices, id_maps

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

    def _search_candidates(
        self, query: np.ndarray, index: Any, id_map: List[int], limit: int
    ) -> List[int]:
        if limit <= 0:
            return []
        if not id_map:
            return []

        k_search = min(limit, len(id_map))
        _, candidate_indices = index.search(query.astype("float32", copy=False), k_search)
        return [id_map[idx] for idx in candidate_indices[0] if idx != -1]

    def _recommend_single(
        self,
        user_id: int,
        league: str,
        query_map: Dict[int, int],
        query_factors: Any,
        candidate_indices: Dict[str, Any],
        candidate_id_map: Dict[str, List[int]],
        pair_builder,
        k: int,
    ) -> List[Tuple[int, float]]:
        allowed_leagues = self._allowed_leagues(league)
        available_leagues = [lvl for lvl in allowed_leagues if lvl in candidate_indices]

        if not available_leagues:
            return []

        query_vector = self._extract_query_vector(user_id, query_map, query_factors)
        if query_vector is None:
            return []

        total_available = sum(len(candidate_id_map.get(lvl, [])) for lvl in available_leagues)
        if total_available == 0:
            return []

        candidate_ids: List[int] = []
        max_candidates = min(total_available, max(k, int(k * self.OVERFETCH_MULTIPLIER)))

        for allowed_league in available_leagues:
            if len(candidate_ids) >= max_candidates:
                break
            index = candidate_indices.get(allowed_league)
            id_map = candidate_id_map.get(allowed_league)
            if index is None or not id_map:
                continue
            remaining = max_candidates - len(candidate_ids)
            if remaining <= 0:
                break
            candidate_ids.extend(
                self._search_candidates(query_vector, index, id_map, remaining)
            )

        candidate_ids = self._dedupe_preserve_order(candidate_ids)
        candidate_ids = candidate_ids[:max_candidates]

        if not candidate_ids:
            return []

        pairs = [pair_builder(user_id, cid) for cid in candidate_ids]
        scores = np.asarray(self.als_model.mutual_score_batch(pairs), dtype=np.float32)

        recommendations = [
            (cid, float(score)) for cid, score in zip(candidate_ids, scores) if score > 0
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
        league: str,
        query_map: Dict[int, int],
        query_factors: Any,
        candidate_indices: Dict[str, Any],
        candidate_id_map: Dict[str, List[int]],
        pair_builder,
        k: int,
    ) -> Dict[int, List[Tuple[int, float]]]:
        results = {uid: [] for uid in user_ids}

        allowed_leagues = self._allowed_leagues(league)
        available_leagues = [lvl for lvl in allowed_leagues if lvl in candidate_indices]

        if not available_leagues:
            return results

        valid_ids, query_matrix = self._collect_query_matrix(user_ids, query_map, query_factors)
        if query_matrix is None:
            return results

        total_available = sum(len(candidate_id_map.get(lvl, [])) for lvl in available_leagues)
        if total_available == 0:
            return results

        per_user_candidates: Dict[int, List[int]] = {uid: [] for uid in valid_ids}
        pairs: List[Tuple[int, int]] = []
        owners: List[int] = []
        candidates: List[int] = []
        max_per_user = min(total_available, max(k, int(k * self.OVERFETCH_MULTIPLIER)))

        for allowed_league in available_leagues:
            index = candidate_indices.get(allowed_league)
            id_map = candidate_id_map.get(allowed_league)
            if index is None or not id_map:
                continue

            k_search = min(max_per_user, len(id_map))
            if k_search == 0:
                continue

            _, candidate_matrix = index.search(query_matrix, k_search)

            for row_idx, uid in enumerate(valid_ids):
                if len(per_user_candidates[uid]) >= max_per_user:
                    continue
                for candidate_idx in candidate_matrix[row_idx]:
                    if candidate_idx == -1:
                        continue
                    candidate_id = id_map[candidate_idx]
                    if candidate_id in per_user_candidates[uid]:
                        continue
                    per_user_candidates[uid].append(candidate_id)
                    pairs.append(pair_builder(uid, candidate_id))
                    owners.append(uid)
                    candidates.append(candidate_id)
                    if len(per_user_candidates[uid]) >= max_per_user:
                        break

        if not pairs:
            return results

        scores = np.asarray(self.als_model.mutual_score_batch(pairs), dtype=np.float32)
        per_user: Dict[int, List[Tuple[int, float]]] = {uid: [] for uid in valid_ids}

        for owner, candidate_id, score in zip(owners, candidates, scores):
            if score > 0:
                per_user.setdefault(owner, []).append((candidate_id, float(score)))

        for uid in valid_ids:
            recs = per_user.get(uid, [])
            recs.sort(key=lambda x: x[1], reverse=True)
            results[uid] = recs[:k]

        return results
