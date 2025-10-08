"""
ELO-based dynamic rating system for the matchmaker library.

This module implements an ELO rating system adapted for dating app matching,
where ratings update based on match outcomes (mutual likes, one-sided likes, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    import cudf  # type: ignore[import]
    import cupy as cp  # type: ignore[import]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "ELO scoring requires RAPIDS (cudf + cupy). Install with: "
        "pip install matchmaker[gpu]"
    ) from exc


@dataclass
class EloConfig:
    """Configuration for the ELO rating system."""
    
    # K-factor: how much ratings change per interaction
    # Higher = more volatile, responds faster to recent performance
    # Lower = more stable, resistant to variance
    k_factor: float = 32.0
    
    # Initial ELO rating for new users
    initial_rating: float = 1200.0
    
    # Minimum and maximum rating bounds
    min_rating: float = 100.0   # Floor to prevent extreme negatives
    max_rating: float = 10000.0  # Ceiling for stability
    
    # Outcome scores for ELO updates
    # In dating apps, ELO should measure DESIRABILITY (being liked), not selectivity
    # So we primarily update based on how others respond to you
    like_reward: float = 1.0       # Someone liked you (you're desirable)
    reject_penalty: float = 0.0    # Someone rejected you (neutral - dating is selective)
    
    # Recency weighting: recent interactions matter more
    recency_halflife_days: Optional[float] = None  # Disabled by default for performance
    
    # Minimum interactions before ELO is considered stable
    min_interactions: int = 10


@dataclass
class EloSummary:
    """Diagnostics for monitoring the ELO computation."""
    
    total_users_scored: int
    avg_rating: float
    median_rating: float
    rating_std: float
    stable_users: int  # Users with >= min_interactions


class EloRatingSystem:
    """
    Compute ELO ratings from interaction history.
    
    Unlike static PageRank or composite scores, ELO ratings:
    - Update after each interaction (dynamic)
    - Reflect actual matching success (not just profile quality)
    - Self-correct over time (good matchers rise, poor matchers fall)
    - Handle new users gracefully (start at initial_rating)
    """
    
    def __init__(self, config: Optional[EloConfig] = None):
        """
        Initialize the ELO rating system.
        
        Args:
            config: Configuration object, uses defaults if None
        """
        self.config = config or EloConfig()
        self.summary_: Optional[EloSummary] = None
    
    def score(
        self,
        interactions: cudf.DataFrame,
        *,
        decider_col: str,
        other_col: str,
        like_col: str,
        timestamp_col: Optional[str] = None,
        gender_col: Optional[str] = None
    ) -> cudf.DataFrame:
        """
        Compute ELO ratings for all users based on interaction history.
        
        This is optimized to minimize iteration by processing interactions in batches
        where possible, but still maintains chronological order for accuracy.
        
        Args:
            interactions: DataFrame with interaction history
            decider_col: Column name for user making the decision
            other_col: Column name for target user
            like_col: Column name for like/dislike (1/0)
            timestamp_col: Column name for timestamps (optional, for chronological processing)
            gender_col: Column name for decider gender (optional, for gender-specific ratings)
            
        Returns:
            DataFrame with columns: user_id, elo_rating, interaction_count, is_stable
        """
        # If gender column provided, compute separate ELO for each gender
        if gender_col is not None:
            return self._score_by_gender(
                interactions, decider_col, other_col, like_col, timestamp_col, gender_col
            )
        
        # Otherwise use single ELO scale (original behavior)
        return self._score_single_scale(
            interactions, decider_col, other_col, like_col, timestamp_col
        )
    
    def _score_by_gender(
        self,
        interactions: cudf.DataFrame,
        decider_col: str,
        other_col: str,
        like_col: str,
        timestamp_col: Optional[str],
        gender_col: str
    ) -> cudf.DataFrame:
        """Compute gender-specific ELO ratings measuring DESIRABILITY.
        
        Key insight: ELO should measure how often you get LIKED, not how selective you are.
        - High ELO = you get liked often when others swipe on you (desirable)
        - Low ELO = you get rejected often when others swipe on you (less desirable)
        - Your own swiping behavior (selectivity) does NOT affect your ELO
        
        This creates separate rating pools for M/F where users are compared only within their gender.
        """
        df = interactions.copy()
        
        # Build gender lookup
        decider_lookup = df[[decider_col, gender_col]].drop_duplicates().rename(
            columns={decider_col: 'user_id'}
        ).to_pandas().set_index('user_id')[gender_col].to_dict()
        
        # Infer "other" gender (opposite for heterosexual matching)
        other_gender_df = df[[other_col, gender_col]].copy()
        other_gender_df[gender_col] = other_gender_df[gender_col].to_pandas().map({'M': 'F', 'F': 'M'})
        other_lookup = other_gender_df.rename(
            columns={other_col: 'user_id'}
        ).drop_duplicates().to_pandas().set_index('user_id')[gender_col].to_dict()
        
        # Combine (decider takes precedence)
        gender_lookup = {**other_lookup, **decider_lookup}
        
        # Process each gender separately
        results = []
        for target_gender in ['M', 'F']:
            # Get all users of this gender
            target_users = sorted([uid for uid, g in gender_lookup.items() if g == target_gender])
            if len(target_users) == 0:
                continue
            
            n_users = len(target_users)
            user_to_idx = {int(user): idx for idx, user in enumerate(target_users)}
            
            # Filter to interactions where someone swiped on this gender
            # (we only care about how this gender gets responded to)
            others_pd = df[other_col].to_pandas()
            target_set = set(target_users)
            mask = others_pd.isin(target_set)
            
            if not mask.any():
                continue
            
            # Get data for these interactions
            filtered_df = df[mask]
            others_filtered = filtered_df[other_col].to_pandas().map(user_to_idx).values
            likes_filtered = filtered_df[like_col].values_host
            
            # GPU arrays
            oth_idx_gpu = cp.asarray(others_filtered, dtype=cp.int32)
            likes_gpu = cp.asarray(likes_filtered, dtype=cp.int32)
            
            # Outcome: 1.0 if liked, 0.0 if rejected
            outcomes = likes_gpu.astype(cp.float32)
            
            # Calculate the baseline like rate for this gender pool
            # Males typically have ~4-5% like rate, females ~45-50%
            # This calibrates the expected scores so ratings center around initial_rating
            baseline_like_rate = float(cp.mean(outcomes))
            
            # Initialize ratings
            ratings = cp.full(n_users, self.config.initial_rating, dtype=cp.float32)
            counts = cp.zeros(n_users, dtype=cp.int32)
            
            # Process in chunks
            n_interactions = len(oth_idx_gpu)
            chunk_size = min(100000, n_interactions)
            n_chunks = (n_interactions + chunk_size - 1) // chunk_size
            
            for chunk_idx in range(n_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, n_interactions)
                
                # Get chunk
                oth_chunk = oth_idx_gpu[start_idx:end_idx]
                outcome_chunk = outcomes[start_idx:end_idx]
                
                # Current ratings for these users
                current_ratings = ratings[oth_chunk]
                
                # Expected score: use the baseline like rate as the neutral point
                # Users with average rating should have expected = baseline_like_rate
                # Higher rated users should have higher expected scores
                avg_rating = cp.mean(ratings)
                rating_diff = current_ratings - avg_rating
                
                # Scale expected score around the baseline
                # At average rating: expected = baseline
                # At +400 rating: expected ≈ baseline + (1-baseline)/10
                # At -400 rating: expected ≈ baseline - baseline/10
                expected = baseline_like_rate + (rating_diff / 400.0) * (1.0 - baseline_like_rate)
                expected = cp.clip(expected, 0.0, 1.0)
                
                # Update: rating increases if liked more than expected
                changes = self.config.k_factor * (outcome_chunk - expected)
                cp.add.at(ratings, oth_chunk, changes)
                cp.add.at(counts, oth_chunk, 1)
                
                # Clamp
                ratings = cp.clip(ratings, self.config.min_rating, self.config.max_rating)
            
            # Build results
            result = cudf.DataFrame({
                'user_id': target_users,
                'elo_rating': cp.asnumpy(ratings),
                'interaction_count': cp.asnumpy(counts),
                'gender': target_gender
            })
            result['is_stable'] = result['interaction_count'] >= self.config.min_interactions
            results.append(result)
        
        # Combine
        if len(results) == 0:
            return self._score_single_scale(interactions, decider_col, other_col, like_col, timestamp_col)
        
        combined = cudf.concat(results, ignore_index=True)
        self._compute_summary(combined)
        return combined
    
    def _score_single_scale(
        self,
        interactions: cudf.DataFrame,
        decider_col: str,
        other_col: str,
        like_col: str,
        timestamp_col: Optional[str]
    ) -> cudf.DataFrame:
        """Compute ELO ratings using a single scale (original implementation)."""
        df = interactions.copy()
        deciders = df[decider_col].astype('int32')
        others = df[other_col].astype('int32')
        likes = df[like_col].astype('int32')

        # Get all unique users
        all_users = cudf.concat([deciders, others]).unique().astype('int32')
        n_users = len(all_users)
        user_to_idx = {int(user): idx for idx, user in enumerate(all_users.to_pandas())}

        # Map user ids to indices for fast lookup
        dec_idx = deciders.to_pandas().map(user_to_idx).values
        oth_idx = others.to_pandas().map(user_to_idx).values

        # Move to GPU
        dec_idx_gpu = cp.asarray(dec_idx, dtype=cp.int32)
        oth_idx_gpu = cp.asarray(oth_idx, dtype=cp.int32)
        likes_gpu = cp.asarray(likes.values_host, dtype=cp.int32)

        # Precompute outcome scores on GPU
        decider_outcomes = cp.where(
            likes_gpu == 1,
            self.config.like_reward,
            self.config.reject_penalty
        ).astype(cp.float32)
        other_outcomes = cp.where(
            likes_gpu == 1,
            self.config.liked_reward,
            self.config.rejected_penalty
        ).astype(cp.float32)

        # Initialize ratings and counts
        ratings = cp.full(n_users, self.config.initial_rating, dtype=cp.float32)
        counts = cp.zeros(n_users, dtype=cp.int32)

        # Iterative batch updates: process in chunks to allow ratings to update
        # This balances speed with accuracy
        n_interactions = len(dec_idx_gpu)
        chunk_size = min(100000, n_interactions)  # Process 100K interactions at a time
        n_chunks = (n_interactions + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, n_interactions)
            
            # Get chunk indices
            dec_chunk = dec_idx_gpu[start_idx:end_idx]
            oth_chunk = oth_idx_gpu[start_idx:end_idx]
            dec_out_chunk = decider_outcomes[start_idx:end_idx]
            oth_out_chunk = other_outcomes[start_idx:end_idx]
            
            # Compute expected scores using current ratings
            dec_ratings = ratings[dec_chunk]
            oth_ratings = ratings[oth_chunk]
            expected_dec = 1.0 / (1.0 + cp.power(10, (oth_ratings - dec_ratings) / 400.0))
            expected_oth = 1.0 / (1.0 + cp.power(10, (dec_ratings - oth_ratings) / 400.0))
            
            # Compute rating changes
            dec_changes = self.config.k_factor * (dec_out_chunk - expected_dec)
            oth_changes = self.config.k_factor * (oth_out_chunk - expected_oth)
            
            # Apply changes using scatter-add
            cp.add.at(ratings, dec_chunk, dec_changes)
            cp.add.at(ratings, oth_chunk, oth_changes)
            cp.add.at(counts, dec_chunk, 1)
            cp.add.at(counts, oth_chunk, 1)
            
            # Clamp ratings to prevent extreme values
            ratings = cp.clip(ratings, self.config.min_rating, self.config.max_rating)

        # Build results DataFrame
        results = cudf.DataFrame({
            'user_id': all_users,
            'elo_rating': cp.asnumpy(ratings),
            'interaction_count': cp.asnumpy(counts)
        })

        # Mark stable users (those with enough interactions)
        results['is_stable'] = results['interaction_count'] >= self.config.min_interactions

        # Compute summary statistics
        self._compute_summary(results)

        return results
    
    def _compute_summary(self, results: cudf.DataFrame) -> None:
        """Compute and store summary statistics."""
        ratings = results['elo_rating']
        
        self.summary_ = EloSummary(
            total_users_scored=len(results),
            avg_rating=float(ratings.mean()),
            median_rating=float(ratings.median()),
            rating_std=float(ratings.std()),
            stable_users=int(results['is_stable'].sum())
        )
