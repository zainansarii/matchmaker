"""
Data loading and preprocessing for the matchmaker library.
"""

import cudf
import cugraph
from typing import Optional


class DataLoader:
    """Handles loading and preprocessing of interaction data."""
    
    def __init__(self):
        self._interactions_df: Optional[cudf.DataFrame] = None
        self._user_df: Optional[cudf.DataFrame] = None
        self._interaction_graph: Optional[cugraph.Graph] = None
        self._metadata = {}
    
    def load_interactions(self, data_path: str, decider_col: str, other_col: str, 
                         like_col: str, timestamp_col: str) -> 'DataLoader':
        """
        Load interaction data and build graph representation.
        
        Args:
            data_path: Path to CSV file containing interaction data
            decider_col: Column name for the user making the decision
            other_col: Column name for the target user
            like_col: Column name for the interaction type (like/dislike)
            timestamp_col: Column name for the timestamp
            
        Returns:
            Self for method chaining
        """
        # Store original column mapping
        self._metadata = {
            'decider_col': decider_col,
            'other_col': other_col, 
            'like_col': like_col,
            'timestamp_col': timestamp_col,
            'data_path': data_path
        }
        
        # Load and preprocess data
        raw_df = cudf.read_csv(data_path)
        self._interactions_df = self._preprocess_interactions(raw_df, decider_col, other_col, like_col, timestamp_col)
        
        # Build derived data structures
        self._build_graph()
        self._build_user_df()
        
        return self
    
    def _preprocess_interactions(self, df: cudf.DataFrame, decider_col: str, other_col: str, 
                               like_col: str, timestamp_col: str) -> cudf.DataFrame:
        """Preprocess raw interaction data."""
        # Select relevant columns without renaming
        interactions = df[[decider_col, other_col, like_col, timestamp_col]].copy()
        
        # Clean and convert timestamp
        interactions = interactions[interactions[timestamp_col].notnull()]
        interactions[timestamp_col] = cudf.to_datetime(interactions[timestamp_col])
        
        return interactions
    
    def _build_graph(self):
        """Build a directed graph from the interactions DataFrame."""
        if self._interactions_df is None:
            raise ValueError("No interactions data available")
            
        graph = cugraph.Graph(directed=True)
        
        # Build graph with interaction type as edge weight
        graph.from_cudf_edgelist(
            self._interactions_df,
            source=self._metadata['decider_col'],
            destination=self._metadata['other_col'],
            edge_attr=self._metadata['like_col'],
            store_transposed=True
        )
        
        self._interaction_graph = graph
    
    def _build_user_df(self):
        """Extract unique users from interactions."""
        if self._interactions_df is None:
            raise ValueError("No interactions data available")
            
        # Get unique users from both decider and other columns
        decider_col = self._metadata['decider_col']
        other_col = self._metadata['other_col']
        
        deciders = self._interactions_df[[decider_col]].rename(columns={decider_col: 'user_id'})
        others = self._interactions_df[[other_col]].rename(columns={other_col: 'user_id'})
        
        # Combine and deduplicate
        all_users = cudf.concat([deciders, others], ignore_index=True)
        self._user_df = all_users.drop_duplicates().reset_index(drop=True)
    
    @property
    def interactions_df(self) -> cudf.DataFrame:
        """Get the processed interactions DataFrame."""
        if self._interactions_df is None:
            raise ValueError("No data loaded. Call load_interactions() first.")
        return self._interactions_df
    
    @property
    def user_df(self) -> cudf.DataFrame:
        """Get the users DataFrame."""
        if self._user_df is None:
            raise ValueError("No data loaded. Call load_interactions() first.")
        return self._user_df
    
    @property
    def interactions_graph(self) -> cugraph.Graph:
        """Get the interaction graph."""
        if self._interaction_graph is None:
            raise ValueError("No graph available. Call load_interactions() first.")
        return self._interaction_graph
    
    @property
    def metadata(self) -> dict:
        """Get metadata about the loaded data."""
        return self._metadata.copy()
    
    def is_loaded(self) -> bool:
        """Check if data has been loaded."""
        return self._interactions_df is not None
    
    def get_user_interactions(self, user_id: int, as_decider: bool = True) -> cudf.DataFrame:
        """
        Get all interactions for a specific user.
        
        Args:
            user_id: The user ID to query
            as_decider: If True, get interactions where user is the decider;
                       if False, get interactions where user is the target
                       
        Returns:
            DataFrame of relevant interactions
        """
        if not self.is_loaded():
            raise ValueError("No data loaded. Call load_interactions() first.")
            
        col_name = self._metadata['decider_col'] if as_decider else self._metadata['other_col']
        return self._interactions_df[self._interactions_df[col_name] == user_id]
