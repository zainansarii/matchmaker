import numpy as np
import cupy as cp
import cudf
from scipy import sparse as sp
from implicit.als import AlternatingLeastSquares
from typing import List, Tuple
import warnings
warnings.filterwarnings("ignore")


class ALSModel:
    def __init__(self, factors=64, regularization=0.01, alpha=1.0, iterations=15):
        
        self.alpha = alpha
        self.model_m2f = AlternatingLeastSquares(
            factors=factors, regularization=regularization, iterations=iterations
        )
        self.model_f2m = AlternatingLeastSquares(
            factors=factors, regularization=regularization, iterations=iterations
        )

        # mappings
        self.male_map, self.female_map = {}, {}
        self.rev_male_map, self.rev_female_map = {}, {}

    # ---------------------- Data Preparation ---------------------- #
    def _join_with_genders(self, interactions, users, decider_col, other_col, gender_col):
        """Join interactions with genders for both decider and other users."""
        df = interactions.merge(
            users.rename(columns={"user_id": decider_col, gender_col: "decider_gender"}),
            on=decider_col,
            how="inner",
        )
        df = df.merge(
            users.rename(columns={"user_id": other_col, gender_col: "other_gender"}),
            on=other_col,
            how="inner",
        )
        return df

    def _filter_active_users(self, df, user_col, min_interactions):
        """Filter users with at least min_interactions."""
        if len(df) == 0:
            raise ValueError(f"No interactions found after gender filtering. Check data for heterosexual pairs.")
        counts = df.groupby(user_col).size().reset_index(name="count")
        active_users = counts[counts["count"] >= min_interactions][user_col]
        filtered_df = df.merge(active_users.to_frame(), on=user_col, how="inner")
        if len(filtered_df) == 0:
            raise ValueError(f"No users with at least {min_interactions} interactions after filtering.")
        return filtered_df

    # ---------------------- Matrix Builders ---------------------- #
    def _build_matrix(self, df, row_col, col_col, like_col):
        """Build sparse CSR matrix from interaction df, using only positive interactions."""
        # Filter to only positive interactions (likes=1)
        # Implicit ALS treats unobserved as negative, so we only include positives
        df_pos = df[df[like_col] > 0].copy()
        
        if len(df_pos) == 0:
            raise ValueError("No positive interactions (likes) found after filtering.")
        
        # Get unique IDs in the same order as they appear
        row_ids = df_pos[row_col].unique().to_pandas().tolist()
        col_ids = df_pos[col_col].unique().to_pandas().tolist()

        if not row_ids or not col_ids:
            raise ValueError("No unique row or column IDs found in positive interactions.")

        row_map = {uid: i for i, uid in enumerate(row_ids)}
        col_map = {uid: i for i, uid in enumerate(col_ids)}
        rev_row_map = {i: uid for uid, i in row_map.items()}
        rev_col_map = {i: uid for uid, i in col_map.items()}

        rows = df_pos[row_col].map(row_map).to_pandas().values
        cols = df_pos[col_col].map(col_map).to_pandas().values
        # For positive-only matrix, confidence is just alpha-weighted count
        vals = (1 + self.alpha * df_pos[like_col].astype("float32")).to_pandas().values

        # Use numpy directly since values are already numpy arrays
        mat = sp.csr_matrix(
            (vals, (rows, cols)),
            shape=(len(row_map), len(col_map)),
        )
        return mat, row_map, col_map, rev_row_map, rev_col_map

    # ---------------------- Training ---------------------- #
    def fit(
        self,
        interactions: cudf.DataFrame,
        users: cudf.DataFrame,
        decider_col: str,
        other_col: str,
        like_col: str,
        gender_col: str,
        min_interactions: int = 2,
    ):
        """
        Train ALS on heterosexual matches and cache factors as CuPy arrays.
        """
        print("🚀 Preparing data...")
        df = self._join_with_genders(interactions, users, decider_col, other_col, gender_col)

        # --- Male → Female --- #
        m2f = df[(df.decider_gender == "M") & (df.other_gender == "F")].copy()
        m2f = self._filter_active_users(m2f, decider_col, min_interactions)

        self.M2F, self.male_map, self.female_map, self.rev_male_map, self.rev_female_map = \
            self._build_matrix(m2f, decider_col, other_col, like_col)

        # --- Female → Male --- #
        f2m = df[(df.decider_gender == "F") & (df.other_gender == "M")].copy()
        f2m = self._filter_active_users(f2m, decider_col, min_interactions)
        self.F2M, self.female_map_f2m, self.male_map_f2m, self.rev_female_map_f2m, self.rev_male_map_f2m = \
            self._build_matrix(f2m, decider_col, other_col, like_col)

        # --- Train models --- #
        print("🎯 Training male→female ALS...")
        self.model_m2f.fit(self.M2F)

        print("🎯 Training female→male ALS...")
        self.model_f2m.fit(self.F2M)

        # Cache factors as CuPy arrays for efficient GPU scoring
        # Convert from implicit's format to numpy first, then to CuPy
        print("🔄 Converting factors to CuPy arrays...")
        
        # Get factors - implicit returns them as properties
        m2f_user = self.model_m2f.user_factors
        m2f_item = self.model_m2f.item_factors
        f2m_user = self.model_f2m.user_factors
        f2m_item = self.model_f2m.item_factors
        
        # Convert to numpy arrays if needed (implicit may return different types)
        if hasattr(m2f_user, 'to_numpy'):
            m2f_user = m2f_user.to_numpy()
        if hasattr(m2f_item, 'to_numpy'):
            m2f_item = m2f_item.to_numpy()
        if hasattr(f2m_user, 'to_numpy'):
            f2m_user = f2m_user.to_numpy()
        if hasattr(f2m_item, 'to_numpy'):
            f2m_item = f2m_item.to_numpy()
        
        # Now convert to CuPy (ensure they're numpy arrays first)
        self.male_factors = cp.asarray(np.asarray(m2f_user, dtype=np.float32))
        self.female_factors = cp.asarray(np.asarray(m2f_item, dtype=np.float32))
        self.female_pref_factors = cp.asarray(np.asarray(f2m_user, dtype=np.float32))
        self.male_attr_factors = cp.asarray(np.asarray(f2m_item, dtype=np.float32))
        
        # Create vectorized lookup arrays for fast batch scoring
        # Instead of dict lookups in loops, we can use numpy array indexing
        self._build_vectorized_maps()

        print(f"✅ Trained M2F ALS with {self.M2F.shape[0]} males × {self.M2F.shape[1]} females")
        print(f"✅ Trained F2M ALS with {self.F2M.shape[0]} females × {self.F2M.shape[1]} males")
        return self
    
    def _build_vectorized_maps(self):
        """Build vectorized lookup arrays for fast batch scoring."""
        # Find max IDs to size the arrays
        max_male_id = max(max(self.male_map.keys()), max(self.male_map_f2m.keys()))
        max_female_id = max(max(self.female_map.keys()), max(self.female_map_f2m.keys()))
        
        # Create lookup arrays: id -> factor_index (use -1 for missing)
        self.male_map_vec = np.full(max_male_id + 1, -1, dtype=np.int32)
        self.female_map_vec = np.full(max_female_id + 1, -1, dtype=np.int32)
        self.male_map_f2m_vec = np.full(max_male_id + 1, -1, dtype=np.int32)
        self.female_map_f2m_vec = np.full(max_female_id + 1, -1, dtype=np.int32)
        
        for user_id, idx in self.male_map.items():
            self.male_map_vec[user_id] = idx
        for user_id, idx in self.female_map.items():
            self.female_map_vec[user_id] = idx
        for user_id, idx in self.male_map_f2m.items():
            self.male_map_f2m_vec[user_id] = idx
        for user_id, idx in self.female_map_f2m.items():
            self.female_map_f2m_vec[user_id] = idx

    # ---------------------- Scoring & Recs ---------------------- #
    def mutual_score(self, male_id, female_id):
        """Calculate bidirectional mutual attraction score between a male and female user."""
        # Convert to int if needed for lookup
        male_id = int(male_id) if not isinstance(male_id, int) else male_id
        female_id = int(female_id) if not isinstance(female_id, int) else female_id
        
        score_total = 0.0
        
        # M2F score: how much does the male like this female?
        if male_id in self.male_map and female_id in self.female_map:
            m_idx = self.male_map[male_id]
            f_idx = self.female_map[female_id]
            score_m2f = float(cp.dot(self.male_factors[m_idx], self.female_factors[f_idx]))
            score_total += score_m2f
        
        # F2M score: how much does the female like this male?
        if female_id in self.female_map_f2m and male_id in self.male_map_f2m:
            f_idx = self.female_map_f2m[female_id]
            m_idx = self.male_map_f2m[male_id]
            score_f2m = float(cp.dot(self.female_pref_factors[f_idx], self.male_attr_factors[m_idx]))
            score_total += score_f2m
        
        # Return None only if we have no information at all
        if score_total == 0.0:
            # Check if we had any valid indices
            has_m2f = male_id in self.male_map and female_id in self.female_map
            has_f2m = female_id in self.female_map_f2m and male_id in self.male_map_f2m
            if not has_m2f and not has_f2m:
                return None
        
        return score_total
    
    def mutual_score_batch(self, pairs: List[Tuple[int, int]]) -> np.ndarray:
        """
        Calculate mutual scores for a batch of (male_id, female_id) pairs - VECTORIZED VERSION.
        
        Uses vectorized array lookups instead of dict iterations for 10-100x speedup.
        
        Args:
            pairs: List of (male_id, female_id) tuples
            
        Returns:
            numpy array of scores (same length as pairs)
        """
        if not pairs:
            return np.array([])
        
        n_pairs = len(pairs)
        
        # Convert pairs to numpy arrays
        pairs_array = np.array(pairs, dtype=np.int32)
        male_ids = pairs_array[:, 0]
        female_ids = pairs_array[:, 1]
        
        scores_gpu = cp.zeros(n_pairs, dtype=cp.float32)
        
        # M2F: Vectorized lookups (no Python loops!)
        # Clip IDs to valid range for array indexing
        male_ids_clipped = np.clip(male_ids, 0, len(self.male_map_vec) - 1)
        female_ids_clipped = np.clip(female_ids, 0, len(self.female_map_vec) - 1)
        
        # Vectorized lookup: get factor indices (-1 if not in map)
        m_factor_idx = self.male_map_vec[male_ids_clipped]
        f_factor_idx = self.female_map_vec[female_ids_clipped]
        
        # Valid pairs are those where both indices >= 0
        m2f_valid = (m_factor_idx >= 0) & (f_factor_idx >= 0)
        
        if m2f_valid.any():
            valid_m_idx = m_factor_idx[m2f_valid]
            valid_f_idx = f_factor_idx[m2f_valid]
            valid_positions = np.where(m2f_valid)[0]
            
            # GPU operations
            m_factors = self.male_factors[valid_m_idx]
            f_factors = self.female_factors[valid_f_idx]
            m2f_scores = cp.sum(m_factors * f_factors, axis=1)
            scores_gpu[valid_positions] += m2f_scores
        
        # F2M: Vectorized lookups
        female_ids_clipped_f2m = np.clip(female_ids, 0, len(self.female_map_f2m_vec) - 1)
        male_ids_clipped_f2m = np.clip(male_ids, 0, len(self.male_map_f2m_vec) - 1)
        
        f_pref_idx = self.female_map_f2m_vec[female_ids_clipped_f2m]
        m_attr_idx = self.male_map_f2m_vec[male_ids_clipped_f2m]
        
        f2m_valid = (f_pref_idx >= 0) & (m_attr_idx >= 0)
        
        if f2m_valid.any():
            valid_f_idx = f_pref_idx[f2m_valid]
            valid_m_idx = m_attr_idx[f2m_valid]
            valid_positions = np.where(f2m_valid)[0]
            
            # GPU operations
            f_factors = self.female_pref_factors[valid_f_idx]
            m_factors = self.male_attr_factors[valid_m_idx]
            f2m_scores = cp.sum(f_factors * m_factors, axis=1)
            scores_gpu[valid_positions] += f2m_scores
        
        # Single GPU->CPU transfer
        return scores_gpu.get()

    def recommend_for_user(self, male_id, N=10, filter_seen=True):
        """Recommend women for a given man, ranked by mutual score (GPU)."""
        # Convert to int if needed for lookup
        male_id = int(male_id) if not isinstance(male_id, int) else male_id
        
        if male_id not in self.male_map:
            return []

        m_idx = self.male_map[male_id]

        # Get recommendations from the M2F model
        # Pass the user's row from the interaction matrix
        recs = self.model_m2f.recommend(
            m_idx,
            self.M2F[m_idx],  # Pass just this user's row
            N=5 * N,
            filter_already_liked_items=filter_seen,
        )

        # recs is a tuple of (item_ids, scores) arrays
        item_ids, _ = recs
        
        # Calculate mutual scores and filter out None values
        scored = []
        for f_idx in item_ids:
            female_id = self.rev_female_map[f_idx]
            score = self.mutual_score(male_id, female_id)
            if score is not None:
                scored.append((female_id, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:N]

    def recommend_batch(self, male_ids, N=10, filter_seen=True):
        """Batch recommend: {male_id: [(female_id, score), ...]} on GPU"""
        # Convert all IDs to int for consistency
        male_ids = [int(mid) if not isinstance(mid, int) else mid for mid in male_ids]
        return {mid: self.recommend_for_user(mid, N=N, filter_seen=filter_seen) for mid in male_ids}


    def get_allowed_ids(self):
        """Return dict with male and female IDs that can be used for recommendations."""
        return {
            'male_ids': list(self.male_map.keys()),
            'female_ids': list(self.female_map.keys())
        }