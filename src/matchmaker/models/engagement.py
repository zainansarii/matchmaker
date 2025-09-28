"""Lightweight engagement scoring utilities for Matchmaker."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

try:
    import cudf  # type: ignore[import]
    import cupy as cp  # type: ignore[import]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Engagement scoring requires RAPIDS (cudf + cupy). Install with: "
        "pip install matchmaker[gpu]"
    ) from exc


_EPSILON = 1e-8
_SECONDS_PER_DAY = 86_400.0


@dataclass
class EngagementConfig:
    """Configuration knobs for the engagement score."""

    activity_weight: float = 0.6
    intentionality_weight: float = 0.4
    min_swipes: int = 3
    recency_halflife_days: Optional[float] = 30.0

    # Quantiles for shaping curves (data-driven instead of hard thresholds)
    activity_low_quantile: float = 0.25
    activity_mid_quantile: float = 0.50
    activity_high_quantile: float = 0.75

    selectivity_low_quantile: float = 0.25
    selectivity_mid_quantile: float = 0.50
    selectivity_high_quantile: float = 0.75

    def weights(self) -> Dict[str, float]:
        total = self.activity_weight + self.intentionality_weight
        if total <= 0:
            raise ValueError("Engagement weights must sum to a positive value.")
        return {
            "activity": self.activity_weight / total,
            "intentionality": self.intentionality_weight / total,
        }


@dataclass
class EngagementSummary:
    """Diagnostics for monitoring the engagement computation."""

    total_users_scored: int
    global_like_rate: float
    activity_reference: Dict[str, float]
    selectivity_reference: Dict[str, float]


class EngagementScorer:
    """Compute intent-aware engagement scores from interaction logs."""

    def __init__(self, config: Optional[EngagementConfig] = None) -> None:
        self.config = config or EngagementConfig()
        self._weights = self.config.weights()
        self.summary_: Optional[EngagementSummary] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def score(
        self,
        interactions: cudf.DataFrame,
        *,
        decider_col: str,
        like_col: str,
        timestamp_col: Optional[str] = None,
    ) -> cudf.DataFrame:
        """Return per-user engagement metrics."""

        self._validate(interactions, decider_col, like_col, timestamp_col)

        working = interactions[[decider_col, like_col]].copy(deep=True)
        working = working.rename(columns={decider_col: "user_id"})
        working[like_col] = working[like_col].fillna(0).astype("float64")

        if timestamp_col is not None:
            weights = self._make_recency_weights(interactions[timestamp_col])
            working["event_weight"] = weights
        else:
            working["event_weight"] = 1.0

        agg = self._aggregate_user_metrics(working, like_col)
        if agg.empty:
            self.summary_ = EngagementSummary(
                total_users_scored=0,
                global_like_rate=0.0,
                activity_reference={},
                selectivity_reference={},
            )
            return agg

        global_like_rate = self._global_like_rate(agg)
        activity_score, activity_ref = self._score_activity(agg)
        intentionality_score, selectivity_ref = self._score_intentionality(
            agg,
            global_like_rate,
        )

        weights = self._weights
        agg["activity_score"] = activity_score
        agg["intentionality_score"] = intentionality_score
        agg["engagement_score"] = (
            activity_score * weights["activity"]
            + intentionality_score * weights["intentionality"]
        ).clip(lower=0.0, upper=1.0)

        self.summary_ = EngagementSummary(
            total_users_scored=int(len(agg)),
            global_like_rate=float(global_like_rate),
            activity_reference=activity_ref,
            selectivity_reference=selectivity_ref,
        )

        return agg

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _validate(
        self,
        interactions: cudf.DataFrame,
        decider_col: str,
        like_col: str,
        timestamp_col: Optional[str],
    ) -> None:
        if not isinstance(interactions, cudf.DataFrame):
            raise TypeError("EngagementScorer expects a cuDF DataFrame.")

        for col in (decider_col, like_col):
            if col not in interactions.columns:
                raise ValueError(f"Missing required column: {col}")

        if timestamp_col and timestamp_col not in interactions.columns:
            raise ValueError(f"Timestamp column '{timestamp_col}' not present.")

    def _make_recency_weights(self, timestamps: cudf.Series) -> cudf.Series:
        if self.config.recency_halflife_days is None:
            return cudf.Series(cp.ones(len(timestamps)), index=timestamps.index)

        ts = cudf.to_datetime(timestamps)
        most_recent = ts.max()
        age_days = (most_recent - ts).dt.total_seconds() / _SECONDS_PER_DAY
        age_days = age_days.fillna(0.0).astype("float64")

        half_life = max(self.config.recency_halflife_days, _EPSILON)
        decay_rate = math.log(2.0) / half_life
        weights = cp.exp(-decay_rate * age_days.to_cupy())
        return cudf.Series(weights, index=timestamps.index)

    def _aggregate_user_metrics(self, df: cudf.DataFrame, like_col: str) -> cudf.DataFrame:
        df = df.copy(deep=True)
        df["weighted_like"] = df[like_col] * df["event_weight"]

        grouped = df.groupby("user_id")
        weighted_swipes = grouped["event_weight"].sum().rename("weighted_swipes")
        weighted_likes = grouped["weighted_like"].sum().rename("weighted_likes")
        total_swipes = grouped.size().rename("total_swipes")
        total_likes = grouped[like_col].sum().rename("total_likes")

        result = cudf.concat(
            [weighted_swipes, weighted_likes, total_swipes, total_likes], axis=1
        ).reset_index()
        result = result.astype(
            {
                "weighted_swipes": "float64",
                "weighted_likes": "float64",
                "total_swipes": "int64",
                "total_likes": "float64",
            }
        )

        result = result[result["total_swipes"] >= self.config.min_swipes]
        if result.empty:
            return result

        result["like_rate_raw"] = (
            result["total_likes"] / result["total_swipes"].clip(lower=1)
        ).fillna(0.0)
        result["like_rate_weighted"] = (
            result["weighted_likes"] / result["weighted_swipes"].clip(lower=_EPSILON)
        ).fillna(0.0)

        return result

    def _global_like_rate(self, df: cudf.DataFrame) -> float:
        likes = float(df["weighted_likes"].sum())
        swipes = float(df["weighted_swipes"].sum()) + _EPSILON
        return max(min(likes / swipes, 1.0), 0.0)

    def _score_activity(
        self, df: cudf.DataFrame
    ) -> tuple[cudf.Series, Dict[str, float]]:
        log_swipes_arr = cp.log(df["weighted_swipes"].to_cupy() + 1.0)
        log_swipes = cudf.Series(log_swipes_arr, index=df.index)

        q_low = float(log_swipes.quantile(self.config.activity_low_quantile))
        q_mid = float(log_swipes.quantile(self.config.activity_mid_quantile))
        q_high = float(log_swipes.quantile(self.config.activity_high_quantile))

        spread = max(q_high - q_low, 0.05)
        scale = max(spread / 1.349, 0.05)

        z = (log_swipes - q_mid) / scale
        scores = cp.exp(-0.5 * cp.square(z.to_cupy()))
        activity = cudf.Series(scores, index=df.index)

        upper_whisker = q_high + 1.5 * (q_high - q_low)
        mask = log_swipes > upper_whisker
        if bool(mask.any()):
            penalty = cp.exp(-cp.abs(z.loc[mask].to_cupy()))
            activity.loc[mask] *= penalty

        reference = {
            "log_swipes_low": q_low,
            "log_swipes_mid": q_mid,
            "log_swipes_high": q_high,
            "upper_whisker": upper_whisker,
        }
        return activity.clip(lower=0.0, upper=1.0), reference

    def _score_intentionality(
        self,
        df: cudf.DataFrame,
        global_like_rate: float,
    ) -> tuple[cudf.Series, Dict[str, float]]:
        median_swipes = max(float(df["total_swipes"].median()), 1.0)
        prior_strength = median_swipes

        alpha_prior = global_like_rate * prior_strength
        beta_prior = (1.0 - global_like_rate) * prior_strength

        smoothed_rate = (
            (df["weighted_likes"] + alpha_prior)
            / (df["weighted_swipes"] + alpha_prior + beta_prior)
        )

        q_low = float(smoothed_rate.quantile(self.config.selectivity_low_quantile))
        q_mid = float(smoothed_rate.quantile(self.config.selectivity_mid_quantile))
        q_high = float(smoothed_rate.quantile(self.config.selectivity_high_quantile))

        spread = max(q_high - q_low, 0.05)
        scale = max(spread / 1.349, 0.05)

        z = (smoothed_rate - q_mid) / scale
        base = cp.exp(-0.5 * cp.square(z.to_cupy()))
        selectivity = cudf.Series(base, index=df.index)

        high_mask = smoothed_rate > q_high
        if bool(high_mask.any()):
            overshoot = smoothed_rate.loc[high_mask] - q_high
            damp = cp.exp(-3.0 * overshoot.to_cupy() / (spread + _EPSILON))
            selectivity.loc[high_mask] *= damp

        confidence = 1.0 - cp.exp(-df["weighted_swipes"].to_cupy() / (prior_strength + _EPSILON))
        confidence_series = cudf.Series(confidence, index=df.index)

        intentionality = (selectivity * confidence_series).clip(lower=0.0, upper=1.0)

        df["like_rate_smoothed"] = smoothed_rate
        df["decision_confidence"] = confidence_series

        reference = {
            "target_ratio": q_mid,
            "ratio_low": q_low,
            "ratio_high": q_high,
            "alpha_prior": alpha_prior,
            "beta_prior": beta_prior,
        }
        return intentionality, reference
