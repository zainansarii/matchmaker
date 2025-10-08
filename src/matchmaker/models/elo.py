"""
ELO-based dynamic rating system for the matchmaker library.

This module implements an ELO rating system adapted for dating app matching,
where ratings update based on match outcomes (mutual likes, one-sided likes, etc.).
"""

from __future__ import annotations

import math
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

    k_factor: float = 32.0
    initial_rating: float = 1200.0
    min_rating: float = 100.0
    max_rating: float = 10000.0
    like_value: float = 1.0
    reject_value: float = 0.0
    expectation_scale: float = 800.0
    chunk_size: int = 100_000
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
            decider_col: Column name for user making the decision (retained for API parity)
            other_col: Column name for the target user whose desirability is rated
            like_col: Column name for like/dislike (1/0)
            timestamp_col: Column name for timestamps (unused in the simplified version)
            gender_col: Column name for target gender (required)
            
        Returns:
            DataFrame with columns: user_id, elo_rating, interaction_count, gender, is_stable
        """
        if gender_col is None:
            raise ValueError("gender_col is required to compute gender-specific ELO ratings.")

        return self._score_by_gender(
            interactions=interactions,
            decider_col=decider_col,
            other_col=other_col,
            like_col=like_col,
            timestamp_col=timestamp_col,
            gender_col=gender_col,
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
        """Compute gender-specific ELO ratings measuring desirability."""
        # Retain parameters for API compatibility
        _ = decider_col
        _ = timestamp_col

        base_cols = [other_col, like_col, gender_col]
        df = interactions[base_cols].dropna(subset=base_cols)

        unique_genders = [
            g for g in df[gender_col].dropna().unique().to_pandas().tolist()
        ]

        results = []
        for target_gender in unique_genders:
            gender_df = df[df[gender_col] == target_gender]
            if gender_df.empty:
                continue

            gender_df = gender_df.dropna(subset=[other_col, like_col])
            if gender_df.empty:
                continue

            codes, categories = gender_df[other_col].astype("int64").factorize()
            if categories.size == 0:
                continue

            codes_int = codes.astype("int32")
            codes_source = getattr(codes_int, "values", codes_int)
            user_idx_gpu = cp.asarray(codes_source, dtype=cp.int32)

            outcomes = gender_df[like_col].astype("float32")
            outcomes_source = getattr(outcomes, "values", outcomes)
            outcomes_gpu = cp.asarray(outcomes_source, dtype=cp.float32)
            outcomes_gpu = cp.where(
                outcomes_gpu > 0,
                self.config.like_value,
                self.config.reject_value,
            ).astype(cp.float32)

            n_interactions = int(user_idx_gpu.size)
            if n_interactions == 0:
                continue

            n_users = len(categories)
            ratings = cp.full(n_users, self.config.initial_rating, dtype=cp.float32)
            counts = cp.zeros(n_users, dtype=cp.int32)

            baseline_like_rate = float(outcomes_gpu.mean()) if n_interactions else 0.0
            scale = self.config.expectation_scale or 1.0

            chunk_size = max(1, min(self.config.chunk_size, n_interactions))
            for start in range(0, n_interactions, chunk_size):
                end = min(start + chunk_size, n_interactions)
                idx_chunk = user_idx_gpu[start:end]
                outcome_chunk = outcomes_gpu[start:end]

                pool_avg = cp.mean(ratings)
                current_ratings = ratings[idx_chunk]
                expected = cp.clip(
                    baseline_like_rate + (current_ratings - pool_avg) / scale,
                    0.0,
                    1.0,
                )

                changes = self.config.k_factor * (outcome_chunk - expected)
                cp.add.at(ratings, idx_chunk, changes)
                cp.add.at(counts, idx_chunk, 1)
                cp.clip(
                    ratings,
                    self.config.min_rating,
                    self.config.max_rating,
                    out=ratings,
                )

            result = cudf.DataFrame(
                {
                    "user_id": cudf.Series(categories).astype("int64"),
                    "elo_rating": cudf.Series(ratings),
                    "interaction_count": cudf.Series(counts),
                }
            )
            result["gender"] = target_gender
            result["is_stable"] = (
                result["interaction_count"] >= self.config.min_interactions
            )
            results.append(result)

        if not results:
            empty = cudf.DataFrame(
                columns=["user_id", "elo_rating", "interaction_count", "gender", "is_stable"]
            )
            self.summary_ = EloSummary(0, 0.0, 0.0, 0.0, 0)
            return empty

        combined = cudf.concat(results, ignore_index=True)
        self._compute_summary(combined)
        return combined

    def _compute_summary(self, results: cudf.DataFrame) -> None:
        """Compute and store summary statistics."""
        if len(results) == 0:
            self.summary_ = EloSummary(0, 0.0, 0.0, 0.0, 0)
            return

        ratings = results['elo_rating']

        avg = float(ratings.mean())
        median = float(ratings.median())
        std = float(ratings.std())

        avg = 0.0 if math.isnan(avg) else avg
        median = 0.0 if math.isnan(median) else median
        std = 0.0 if math.isnan(std) else std

        self.summary_ = EloSummary(
            total_users_scored=len(results),
            avg_rating=avg,
            median_rating=median,
            rating_std=std,
            stable_users=int(results['is_stable'].sum())
        )
