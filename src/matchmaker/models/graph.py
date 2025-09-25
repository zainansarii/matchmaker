"""
Graph-based recommendation models for the matchmaker library.
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
    
    def get_like_stats(self) -> cudf.DataFrame:
        """
        Calculate like statistics for users based on interactions.
        
        Returns:
            DataFrame containing user_id, likes_received, likes_given, and like_ratio
        """
        if not self.is_built():
            raise ValueError("Graph not built. Call build_graph() first.")
            
        decider_col = self._metadata['decider_col']
        other_col = self._metadata['other_col']
        like_col = self._metadata['like_col']
        
        # Likes received per user
        likes_received = self._interactions_df.groupby(other_col)[like_col].sum().reset_index()
        likes_received = likes_received.rename(columns={other_col: 'user_id', like_col: 'likes_received'})

        # Likes given per user
        likes_given = self._interactions_df.groupby(decider_col)[like_col].sum().reset_index()
        likes_given = likes_given.rename(columns={decider_col: 'user_id', like_col: 'likes_given'})

        # Merge like stats together
        like_stats_df = likes_received.merge(likes_given, on='user_id', how='outer')

        # Fill NaNs with 0
        like_stats_df = like_stats_df.fillna(0)

        # Compute like ratio (add small epsilon to avoid div by 0)
        like_stats_df['like_ratio'] = like_stats_df['likes_received'] / (like_stats_df['likes_given'] + 1e-6)

        return like_stats_df[['user_id', 'likes_received', 'likes_given', 'like_ratio']]
    
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
