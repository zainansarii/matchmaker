"""
FAISS-based recommendation system for large-scale matching.

This module provides GPU-accelerated approximate nearest neighbor search
filtered by league for scalable recommendations to millions of users.
"""

import numpy as np
import cupy as cp
import cudf
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import warnings

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError as e:
    FAISS_AVAILABLE = False
    warnings.warn(f"FAISS not available: {e}. Install with: conda install -c pytorch faiss-gpu")
    faiss = None  # type: ignore


class LeagueFilteredRecommender:
    """
    FAISS-based recommender that filters candidates by league before ranking by mutual score.
    
    For dating apps, this ensures users are matched within their attractiveness tier
    while leveraging fast ANN search for scalability.
    """
    
    def __init__(self, als_model, user_df: pd.DataFrame, use_gpu: bool = True):
        """
        Initialize recommender with ALS model and user metadata.
        
        Args:
            als_model: Trained ALSModel with male/female factors
            user_df: DataFrame with columns [user_id, gender, league]
            use_gpu: Whether to use GPU for FAISS (requires faiss-gpu)
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is required. Install with: conda install -c pytorch faiss-gpu")
        
        self.als_model = als_model
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        
        # Convert to pandas if needed
        if isinstance(user_df, cudf.DataFrame):
            user_df = user_df.to_pandas()
        
        # Separate by gender and league
        self._build_league_indices(user_df)
        
    def _build_league_indices(self, user_df: pd.DataFrame):
        """Build FAISS indices for each gender and league combination."""
        
        # Extract male and female users
        males_df = user_df[user_df['gender'] == 'M'].copy()
        females_df = user_df[user_df['gender'] == 'F'].copy()
        
        # Initialize storage for indices
        self.male_indices = {}  # league -> FAISS index
        self.male_id_map = {}   # league -> [user_ids]
        self.female_indices = {}
        self.female_id_map = {}
        
        print(f"Building FAISS indices for league-filtered search...")
        
        # Build indices for each league
        leagues = ['Bronze', 'Silver', 'Gold', 'Platinum', 'Diamond']
        
        for league in leagues:
            # Males in this league
            league_males = males_df[males_df['league'] == league]
            if len(league_males) > 0:
                # Get IDs that are in EITHER M2F or F2M model
                valid_ids = [uid for uid in league_males['user_id'].values 
                            if uid in self.als_model.male_map or uid in self.als_model.male_map_f2m]
                
                if len(valid_ids) > 0:
                    # Get factor embeddings for these males
                    # Prefer M2F model (male_map), fall back to F2M (male_map_f2m)
                    factors_list = []
                    for uid in valid_ids:
                        if uid in self.als_model.male_map:
                            idx = self.als_model.male_map[uid]
                            factors_list.append(self.als_model.male_factors[idx].get())
                        else:
                            # Use male_attr_factors from F2M model
                            idx = self.als_model.male_map_f2m[uid]
                            factors_list.append(self.als_model.male_attr_factors[idx].get())
                    
                    factors = np.vstack(factors_list)
                    
                    # Build FAISS index
                    self.male_indices[league] = self._create_index(factors)
                    self.male_id_map[league] = valid_ids
                    print(f"  {league} males: {len(valid_ids)} users")
            
            # Females in this league  
            league_females = females_df[females_df['league'] == league]
            if len(league_females) > 0:
                # Get IDs that are in EITHER M2F or F2M model
                valid_ids = [uid for uid in league_females['user_id'].values 
                            if uid in self.als_model.female_map or uid in self.als_model.female_map_f2m]
                
                if len(valid_ids) > 0:
                    # Get factor embeddings for these females
                    # Prefer M2F model (female_map), fall back to F2M (female_map_f2m)
                    factors_list = []
                    for uid in valid_ids:
                        if uid in self.als_model.female_map:
                            idx = self.als_model.female_map[uid]
                            factors_list.append(self.als_model.female_factors[idx].get())
                        else:
                            # Use female_pref_factors from F2M model
                            idx = self.als_model.female_map_f2m[uid]
                            factors_list.append(self.als_model.female_pref_factors[idx].get())
                    
                    factors = np.vstack(factors_list)
                    
                    # Build FAISS index
                    self.female_indices[league] = self._create_index(factors)
                    self.female_id_map[league] = valid_ids
                    print(f"  {league} females: {len(valid_ids)} users")
        
        print(f"✅ FAISS indices built for {len(self.male_indices)} male leagues, {len(self.female_indices)} female leagues")
    
    def _create_index(self, factors: np.ndarray) -> Any:
        """Create FAISS index from factor embeddings."""
        d = factors.shape[1]  # dimensionality
        
        # For smaller datasets, use exact search (IndexFlatIP for inner product)
        # For large datasets (>100k), use approximate search (IndexIVFFlat)
        n = factors.shape[0]
        
        # Use CPU for small indices to save GPU memory
        use_gpu_for_this_index = self.use_gpu and n > 5000
        
        if n < 100000:
            # Exact search using inner product (cosine similarity on normalized vectors)
            index = faiss.IndexFlatIP(d)
        else:
            # Approximate search with IVF (Inverted File Index)
            nlist = min(int(np.sqrt(n)), 4096)  # number of clusters
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(factors.astype('float32'))
            index.nprobe = min(32, nlist)  # number of clusters to search
        
        # Move to GPU only for larger indices
        if use_gpu_for_this_index:
            try:
                res = faiss.StandardGpuResources()
                # Limit temp memory to 256MB per index
                res.setTempMemory(256 * 1024 * 1024)
                index = faiss.index_cpu_to_gpu(res, 0, index)
            except Exception as e:
                # Fall back to CPU if GPU allocation fails
                print(f"    Warning: GPU allocation failed for {n} vectors, using CPU: {e}")
        
        # Add vectors
        index.add(factors.astype('float32'))
        return index
    
    def recommend_for_male(self, male_id: int, league: str, k: int = 1000) -> List[Tuple[int, float]]:
        """
        Recommend top-k females in the same league for a male user.
        
        Args:
            male_id: Male user ID
            league: User's league (Bronze/Silver/Gold/Platinum/Diamond)
            k: Number of recommendations to return
            
        Returns:
            List of (female_id, mutual_score) tuples, sorted by score descending
        """
        if male_id not in self.als_model.male_map:
            return []
        
        if league not in self.female_indices:
            return []  # No females in this league
        
        # Get male's factor embedding
        m_idx = self.als_model.male_map[male_id]
        male_factor = self.als_model.male_factors[m_idx].get().reshape(1, -1)
        
        # Search FAISS index for top-k females in same league
        index = self.female_indices[league]
        k_search = min(k, len(self.female_id_map[league]))
        
        distances, indices = index.search(male_factor.astype('float32'), k_search)
        
        # Map FAISS indices back to user IDs and compute mutual scores in batch
        candidate_ids = []
        for idx in indices[0]:
            if idx == -1:  # FAISS returns -1 for empty results
                continue
            female_id = self.female_id_map[league][idx]
            candidate_ids.append(female_id)
        
        # Batch compute mutual scores
        if candidate_ids:
            pairs = [(male_id, fid) for fid in candidate_ids]
            mutual_scores = self.als_model.mutual_score_batch(pairs)
            
            recommendations = [(fid, float(score)) for fid, score in zip(candidate_ids, mutual_scores) if score > 0]
        else:
            recommendations = []
        
        # Sort by mutual score descending
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:k]
    
    def recommend_for_female(self, female_id: int, league: str, k: int = 1000) -> List[Tuple[int, float]]:
        """
        Recommend top-k males in the same league for a female user.
        
        Args:
            female_id: Female user ID
            league: User's league (Bronze/Silver/Gold/Platinum/Diamond)
            k: Number of recommendations to return
            
        Returns:
            List of (male_id, mutual_score) tuples, sorted by score descending
        """
        if female_id not in self.als_model.female_map_f2m:
            return []
        
        if league not in self.male_indices:
            return []  # No males in this league
        
        # Get female's preference factor embedding (from F2M model)
        f_idx = self.als_model.female_map_f2m[female_id]
        female_factor = self.als_model.female_pref_factors[f_idx].get().reshape(1, -1)
        
        # Search FAISS index for top-k males in same league
        index = self.male_indices[league]
        k_search = min(k, len(self.male_id_map[league]))
        
        distances, indices = index.search(female_factor.astype('float32'), k_search)
        
        # Map FAISS indices back to user IDs and compute mutual scores in batch
        candidate_ids = []
        for idx in indices[0]:
            if idx == -1:
                continue
            male_id = self.male_id_map[league][idx]
            candidate_ids.append(male_id)
        
        # Batch compute mutual scores
        if candidate_ids:
            pairs = [(mid, female_id) for mid in candidate_ids]
            mutual_scores = self.als_model.mutual_score_batch(pairs)
            
            recommendations = [(mid, float(score)) for mid, score in zip(candidate_ids, mutual_scores) if score > 0]
        else:
            recommendations = []
        
        # Sort by mutual score descending
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:k]
    
    def recommend_batch(self, user_ids: List[int], user_metadata: Dict[int, Dict[str, str]], 
                       k: int = 1000) -> Dict[int, List[Tuple[int, float]]]:
        """
        Batch recommend for multiple users with optimized FAISS batch search.
        
        Args:
            user_ids: List of user IDs
            user_metadata: Dict mapping user_id -> {'gender': 'M'/'F', 'league': 'Gold', ...}
            k: Number of recommendations per user
            
        Returns:
            Dict mapping user_id -> [(candidate_id, mutual_score), ...]
        """
        results = {}
        
        # Group users by (gender, league) for batched FAISS queries
        groups = {}  # (gender, league) -> [user_ids]
        for uid in user_ids:
            if uid not in user_metadata:
                results[uid] = []
                continue
            
            meta = user_metadata[uid]
            gender = meta.get('gender')
            league = meta.get('league')
            key = (gender, league)
            
            if key not in groups:
                groups[key] = []
            groups[key].append(uid)
        
        # Process each group with batch FAISS search
        for (gender, league), group_uids in groups.items():
            if gender == 'M':
                group_results = self._batch_recommend_males(group_uids, league, k)
            elif gender == 'F':
                group_results = self._batch_recommend_females(group_uids, league, k)
            else:
                group_results = {uid: [] for uid in group_uids}
            
            results.update(group_results)
        
        return results
    
    def _batch_recommend_males(self, male_ids: List[int], league: str, k: int) -> Dict[int, List[Tuple[int, float]]]:
        """Batch recommend females for multiple males in the same league."""
        if league not in self.female_indices:
            return {mid: [] for mid in male_ids}
        
        # Filter to valid males and get their factors
        valid_pairs = []  # (male_id, factor_array)
        for mid in male_ids:
            if mid in self.als_model.male_map:
                m_idx = self.als_model.male_map[mid]
                male_factor = self.als_model.male_factors[m_idx].get()
                valid_pairs.append((mid, male_factor))
        
        if not valid_pairs:
            return {mid: [] for mid in male_ids}
        
        # Stack all male factors into a single matrix for batch search
        male_ids_ordered, factors_list = zip(*valid_pairs)
        query_matrix = np.vstack(factors_list).astype('float32')
        
        # Batch FAISS search - ONE call for all males
        index = self.female_indices[league]
        k_search = min(k, len(self.female_id_map[league]))
        distances, indices = index.search(query_matrix, k_search)
        
        # Collect ALL pairs for batch mutual scoring
        all_pairs = []
        pair_to_user = []  # Track which user each pair belongs to
        
        for i, mid in enumerate(male_ids_ordered):
            for idx in indices[i]:
                if idx == -1:
                    continue
                female_id = self.female_id_map[league][idx]
                all_pairs.append((mid, female_id))
                pair_to_user.append(i)  # Track user index
        
        # Single batch mutual score computation for ALL pairs
        if all_pairs:
            all_scores = self.als_model.mutual_score_batch(all_pairs)
            
            # Group results by user
            user_recommendations = {mid: [] for mid in male_ids_ordered}
            for pair_idx, ((mid, fid), score) in enumerate(zip(all_pairs, all_scores)):
                if score > 0:
                    user_recommendations[mid].append((fid, float(score)))
            
            # Sort each user's recommendations
            results = {}
            for mid in male_ids_ordered:
                recs = user_recommendations[mid]
                recs.sort(key=lambda x: x[1], reverse=True)
                results[mid] = recs[:k]
        else:
            results = {mid: [] for mid in male_ids_ordered}
        
        # Add empty results for invalid males
        for mid in male_ids:
            if mid not in results:
                results[mid] = []
        
        return results
    
    def _batch_recommend_females(self, female_ids: List[int], league: str, k: int) -> Dict[int, List[Tuple[int, float]]]:
        """Batch recommend males for multiple females in the same league."""
        if league not in self.male_indices:
            return {fid: [] for fid in female_ids}
        
        # Filter to valid females and get their factors
        valid_pairs = []  # (female_id, factor_array)
        for fid in female_ids:
            if fid in self.als_model.female_map_f2m:
                f_idx = self.als_model.female_map_f2m[fid]
                female_factor = self.als_model.female_pref_factors[f_idx].get()
                valid_pairs.append((fid, female_factor))
        
        if not valid_pairs:
            return {fid: [] for fid in female_ids}
        
        # Stack all female factors into a single matrix for batch search
        female_ids_ordered, factors_list = zip(*valid_pairs)
        query_matrix = np.vstack(factors_list).astype('float32')
        
        # Batch FAISS search - ONE call for all females
        index = self.male_indices[league]
        k_search = min(k, len(self.male_id_map[league]))
        distances, indices = index.search(query_matrix, k_search)
        
        # Collect ALL pairs for batch mutual scoring
        all_pairs = []
        
        for i, fid in enumerate(female_ids_ordered):
            for idx in indices[i]:
                if idx == -1:
                    continue
                male_id = self.male_id_map[league][idx]
                all_pairs.append((male_id, fid))
        
        # Single batch mutual score computation for ALL pairs
        if all_pairs:
            all_scores = self.als_model.mutual_score_batch(all_pairs)
            
            # Group results by female user
            user_recommendations = {fid: [] for fid in female_ids_ordered}
            for (mid, fid), score in zip(all_pairs, all_scores):
                if score > 0:
                    user_recommendations[fid].append((mid, float(score)))
            
            # Sort each user's recommendations
            results = {}
            for fid in female_ids_ordered:
                recs = user_recommendations[fid]
                recs.sort(key=lambda x: x[1], reverse=True)
                results[fid] = recs[:k]
        else:
            results = {fid: [] for fid in female_ids_ordered}
        
        # Add empty results for invalid females
        for fid in female_ids:
            if fid not in results:
                results[fid] = []
        
        return results
