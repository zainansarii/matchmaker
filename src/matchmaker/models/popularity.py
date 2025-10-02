"""
Popularity-based recommendation models for the matchmaker library.
Includes graph-based metrics and standalone popularity functions.
"""

import cugraph
import cudf
from typing import Optional


class InteractionGraph:
    """Handles graph construction and graph-based metrics computation."""
    
    def __init__(self):
        self._graph: Optional[cugraph.Graph] = None
        self._interactions_df: Optional[cudf.DataFrame] = None
        self._metadata: dict = {}
    
    def build_graph(self, interactions_df: cudf.DataFrame, 
                   decider_col: str, other_col: str, like_col: str) -> 'InteractionGraph':
        """
        Build a directed graph from interactions DataFrame.
        
        Args:
            interactions_df: DataFrame containing user interactions
            decider_col: Column name for the user making the decision
            other_col: Column name for the target user
            like_col: Column name for the interaction type (like/dislike)
            
        Returns:
            Self for method chaining
        """
        self._interactions_df = interactions_df
        self._metadata = {
            'decider_col': decider_col,
            'other_col': other_col,
            'like_col': like_col
        }
        
        # Build directed graph
        self._graph = cugraph.Graph(directed=True)
        self._graph.from_cudf_edgelist(
            interactions_df,
            source=decider_col,
            destination=other_col,
            edge_attr=like_col,
            store_transposed=True
        )
        
        return self
    
    @property
    def graph(self) -> cugraph.Graph:
        """Get the interaction graph."""
        if self._graph is None:
            raise ValueError("Graph not built. Call build_graph() first.")
        return self._graph
    
    def is_built(self) -> bool:
        """Check if graph has been built."""
        return self._graph is not None
    
    def get_pagerank(self, alpha: float = 0.85, max_iter: int = 100, 
                    tol: float = 1e-06) -> cudf.DataFrame:
        """
        Calculate PageRank scores for users in the graph.
        
        Args:
            alpha: Damping factor for PageRank
            max_iter: Maximum number of iterations
            tol: Tolerance for convergence
            
        Returns:
            DataFrame containing user_id and pagerank scores
        """
        if not self.is_built():
            raise ValueError("Graph not built. Call build_graph() first.")
            
        pagerank_df = cugraph.pagerank(self._graph, alpha=alpha, max_iter=max_iter, tol=tol)
        pagerank_df = pagerank_df.rename(columns={'vertex': 'user_id'})
        return pagerank_df
    
    def get_degree_centrality(self) -> cudf.DataFrame:
        """
        Calculate degree centrality for users in the graph.
        
        Returns:
            DataFrame containing user_id and degree centrality scores
        """
        if not self.is_built():
            raise ValueError("Graph not built. Call build_graph() first.")
            
        degree_df = cugraph.degree_centrality(self._graph)
        degree_df = degree_df.rename(columns={'vertex': 'user_id'})
        return degree_df
    
    def get_betweenness_centrality(self, k: Optional[int] = None) -> cudf.DataFrame:
        """
        Calculate betweenness centrality for users in the graph.
        
        Args:
            k: Number of vertices to use for approximation (None for exact)
            
        Returns:
            DataFrame containing user_id and betweenness centrality scores
        """
        if not self.is_built():
            raise ValueError("Graph not built. Call build_graph() first.")
            
        betweenness_df = cugraph.betweenness_centrality(self._graph, k=k)
        betweenness_df = betweenness_df.rename(columns={'vertex': 'user_id'})
        return betweenness_df
    
    def get_node_count(self) -> int:
        """Get the number of nodes in the graph."""
        if not self.is_built():
            raise ValueError("Graph not built. Call build_graph() first.")
        return self._graph.number_of_vertices()
    
    def get_edge_count(self) -> int:
        """Get the number of edges in the graph."""
        if not self.is_built():
            raise ValueError("Graph not built. Call build_graph() first.")
        return self._graph.number_of_edges()


# Constants for popularity calculations
_EPSILON = 1e-8
_SECONDS_PER_DAY = 86_400.0


# Standalone popularity functions
def _calculate_recency_weights(timestamps: cudf.Series, recency_halflife_days: float) -> cudf.Series:
    """
    Calculate recency weights for interactions based on timestamp decay.
    
    Args:
        timestamps: Series of interaction timestamps
        recency_halflife_days: Half-life for recency decay in days
        
    Returns:
        Series of weights (more recent interactions have higher weights)
    """
    import math
    import cupy as cp
    
    if recency_halflife_days <= 0:
        return cudf.Series(cp.ones(len(timestamps)), index=timestamps.index)
    
    ts = cudf.to_datetime(timestamps)
    most_recent = ts.max()
    age_days = (most_recent - ts).dt.total_seconds() / _SECONDS_PER_DAY
    age_days = age_days.fillna(0.0).astype("float64")
    
    decay_rate = math.log(2.0) / max(recency_halflife_days, _EPSILON)
    weights = cp.exp(-decay_rate * age_days.to_cupy())
    return cudf.Series(weights, index=timestamps.index)


def _aggregate_popularity_metrics(df: cudf.DataFrame, like_col: str, min_swipes: int) -> cudf.DataFrame:
    """
    Aggregate popularity metrics per user from weighted interaction data.
    Optimized for performance using two-stage approach.
    
    Args:
        df: DataFrame with user_id, like_col, and event_weight columns
        like_col: Column name for the interaction type (like/dislike)
        min_swipes: Minimum swipes required to include user
        
    Returns:
        DataFrame with aggregated metrics per user
    """
    # Stage 1: Fast count aggregation using value_counts
    user_counts = df['user_id'].value_counts().reset_index()
    user_counts.columns = ['user_id', 'swipes_received']
    
    # Early filter by minimum swipes to reduce data for weighted calculations
    valid_users = user_counts[user_counts['swipes_received'] >= min_swipes]['user_id']
    
    if len(valid_users) == 0:
        return cudf.DataFrame(columns=['user_id', 'likes_received', 'swipes_received', 
                                     'weighted_likes_received', 'weighted_swipes_received'])
    
    # Stage 2: Weighted aggregation only on valid users
    df['weighted_like'] = df[like_col] * df['event_weight']
    df_filtered = df[df['user_id'].isin(valid_users)]
    
    weighted_stats = df_filtered.groupby('user_id', sort=False).agg({
        'event_weight': 'sum',
        'weighted_like': 'sum', 
        like_col: 'sum'
    }).reset_index()
    
    weighted_stats.columns = ['user_id', 'weighted_swipes_received', 'weighted_likes_received', 'likes_received']
    
    # Merge count and weighted stats
    result = user_counts[user_counts['swipes_received'] >= min_swipes].merge(
        weighted_stats, on='user_id', how='inner'
    )
    
    # Convert to proper dtypes
    result = result.astype({
        'weighted_swipes_received': 'float64',
        'weighted_likes_received': 'float64', 
        'swipes_received': 'int64',
        'likes_received': 'float64',
    })
    
    return result


def _calculate_raw_rates(df: cudf.DataFrame) -> None:
    """
    Calculate raw and weighted in-like rates in-place for better performance.
    
    Args:
        df: DataFrame with aggregated metrics (modified in-place)
    """
    # Calculate raw in-like rate
    df['in_like_rate_raw'] = (
        df['likes_received'] / df['swipes_received'].clip(1)
    ).fillna(0.0)
    
    # Calculate weighted in-like rate  
    df['in_like_rate_weighted'] = (
        df['weighted_likes_received'] / df['weighted_swipes_received'].clip(_EPSILON)
    ).fillna(0.0)


def _calculate_global_popularity_rate(df: cudf.DataFrame) -> float:
    """
    Calculate global popularity rate for use as Bayesian prior.
    
    Args:
        df: DataFrame with weighted metrics
        
    Returns:
        Global in-like rate as float
    """
    # Use GPU operations then only convert final result to CPU
    total_weighted_likes = df['weighted_likes_received'].sum()
    total_weighted_swipes = df['weighted_swipes_received'].sum() + _EPSILON
    rate = total_weighted_likes / total_weighted_swipes
    # Only convert final scalar to float - use min/max instead of clip for cuDF
    return float(max(min(rate, 1.0), 0.0))


def _apply_bayesian_smoothing(df: cudf.DataFrame, global_rate: float) -> None:
    """
    Apply Bayesian smoothing to popularity rates for more robust scores in-place.
    
    Args:
        df: DataFrame with raw rates (modified in-place)
        global_rate: Global popularity rate to use as prior
    """
    # Calculate prior strength based on median swipes - keep on GPU
    median_swipes = df['swipes_received'].median()
    median_swipes = median_swipes.clip(1.0)
    prior_strength = float(median_swipes)  # Only convert final scalar
    
    alpha_prior = global_rate * prior_strength
    beta_prior = (1.0 - global_rate) * prior_strength
    
    # Smoothed popularity rate using Bayesian approach - all GPU operations
    df['in_like_rate_smoothed'] = (
        (df['weighted_likes_received'] + alpha_prior) /
        (df['weighted_swipes_received'] + alpha_prior + beta_prior)
    )
    
    # Calculate confidence based on number of interactions - use cupy for vectorized exp
    import cupy as cp
    confidence_arg = -df['weighted_swipes_received'] / (prior_strength + _EPSILON)
    confidence_values = 1.0 - cp.exp(confidence_arg.to_cupy())
    df['popularity_confidence'] = cudf.Series(confidence_values, index=df.index)


def _calculate_final_popularity_score(df: cudf.DataFrame) -> None:
    """
    Calculate final popularity score combining rate, confidence, and volume in-place.
    
    Args:
        df: DataFrame with smoothed rates and confidence (modified in-place)
    """
    # Final popularity score: smoothed rate weighted by confidence and volume
    # Use cupy for square root to ensure GPU acceleration
    import cupy as cp
    likes_sqrt = cp.sqrt(df['likes_received'].to_cupy())
    likes_sqrt_series = cudf.Series(likes_sqrt, index=df.index)
    
    df['popularity_score'] = (
        df['in_like_rate_smoothed'] * 
        df['popularity_confidence'] * 
        likes_sqrt_series
    ).clip(0.0, 1.0)


def get_like_stats(interactions_df: cudf.DataFrame, decider_col: str, other_col: str, like_col: str, 
                  timestamp_col: Optional[str] = None, recency_halflife_days: float = 30.0,
                  min_swipes: int = 3) -> cudf.DataFrame:
    """
    Calculate comprehensive like statistics focusing on in-like rates (when users are swiped on).
    This mirrors the engagement module's out-like rate focus but for incoming interactions.
    Includes recency weighting and Bayesian smoothing for robust popularity scores.
    
    Args:
        interactions_df: DataFrame containing user interactions
        decider_col: Column name for the user making the decision
        other_col: Column name for the target user
        like_col: Column name for the interaction type (like/dislike)
        timestamp_col: Column name for timestamps (optional, enables recency weighting)
        recency_halflife_days: Half-life for recency decay in days
        min_swipes: Minimum swipes required to include user in results
        
    Returns:
        DataFrame containing user_id, likes_received, swipes_received, weighted_likes_received,
        weighted_swipes_received, in_like_rate_raw, in_like_rate_smoothed, and popularity_score
    """
    # Prepare data focused on incoming interactions (when users are swiped on)
    # Minimize copies by selecting columns efficiently
    columns_needed = [decider_col, other_col, like_col]
    if timestamp_col:
        columns_needed.append(timestamp_col)
    
    df = interactions_df[columns_needed].copy()
    
    # Apply recency weighting if timestamps are available
    if timestamp_col:
        df['event_weight'] = _calculate_recency_weights(df[timestamp_col], recency_halflife_days)
        # Drop timestamp column to save memory
        df = df.drop(columns=[timestamp_col])
    else:
        df['event_weight'] = 1.0
    
    # Focus on incoming interactions - rename other_col to user_id for aggregation
    df = df.rename(columns={other_col: 'user_id'})
    
    # Aggregate metrics per user
    result = _aggregate_popularity_metrics(df, like_col, min_swipes)
    if result.empty:
        return result
    
    # Calculate raw rates (in-place modifications to avoid copies)
    _calculate_raw_rates(result)
    
    # Calculate global rate for Bayesian prior
    global_rate = _calculate_global_popularity_rate(result)
    
    # Apply Bayesian smoothing (in-place modifications)
    _apply_bayesian_smoothing(result, global_rate)
    
    # Calculate final popularity score (in-place modifications)
    _calculate_final_popularity_score(result)
    
    return result[['user_id', 'likes_received', 'swipes_received', 'weighted_likes_received', 
                  'weighted_swipes_received', 'in_like_rate_raw', 'in_like_rate_weighted',
                  'in_like_rate_smoothed', 'popularity_confidence', 'popularity_score']]


def assign_balanced_leagues(
    user_df: cudf.DataFrame,
    *,
    pagerank_col: str = 'pagerank',
    gender_col: str = 'gender',
    quantiles = (0.0, 0.3, 0.5, 0.7, 0.9, 1.0),
    labels = ("Bronze", "Silver", "Gold", "Platinum", "Diamond"),
) -> cudf.DataFrame:
    """
    Assign league tiers based on PageRank, preserving gender proportions within each tier.

    Expects user_df to contain columns: 'user_id', pagerank_col, gender_col.
    Returns a cudf.DataFrame with columns ['user_id', 'league'] suitable for merging.
    """
    # Filter to rows with both pagerank and gender
    cols = ['user_id', pagerank_col, gender_col]
    for c in cols:
        if c not in user_df.columns:
            raise ValueError(f"assign_balanced_leagues requires column: {c}")

    df = user_df[cols].dropna(subset=[pagerank_col, gender_col])
    if len(df) == 0:
        return cudf.DataFrame({'user_id': [], 'league': []})

    # Convert minimal frame to pandas for qcut; this is small (~users only)
    import pandas as pd  # local import to avoid global dependency
    pdf = df.to_pandas()

    parts = []
    for g in pdf[gender_col].dropna().unique().tolist():
        sub = pdf[pdf[gender_col] == g].copy()
        if len(sub) < 2:
            # Not enough records to form bins
            sub['league'] = labels[0]
            parts.append(sub[['user_id', 'league']])
            continue

        try:
            # First, compute quantile-based bins without labels so pandas decides actual bin edges
            cats = pd.qcut(
                sub[pagerank_col],
                q=quantiles,
                labels=None,
                duplicates='drop'
            )

            # Determine actual number of bins created (can be fewer due to duplicate edges)
            num_bins = len(cats.cat.categories)
            if num_bins <= 0:
                sub['league'] = labels[0]
            else:
                # Slice labels to the exact number of bins and rename categories accordingly
                sliced_labels = list(labels)[:num_bins]
                sub['league'] = cats.cat.rename_categories(sliced_labels)
        except Exception:
            # Fallback: assign everyone to base tier for this gender on any unexpected error
            sub['league'] = labels[0]

        parts.append(sub[['user_id', 'league']])

    if not parts:
        raise ValueError("No valid genders found in data for league assignment")

    result_pdf = pd.concat(parts, axis=0, ignore_index=True)
    result_cudf = cudf.from_pandas(result_pdf[['user_id', 'league']])
    return result_cudf



