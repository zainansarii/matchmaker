import pandas as pd
import cugraph
from .data.loader import DataLoader
from .models.popularity import get_pagerank, get_like_stats

class MatchingEngine:
  """High level API for performing matching tasks."""
  
  def __init__(self):
    self.data_loader = DataLoader()
    self.interactions_df = None
    self.interactions_graph = None
    self.user_df = None

  def load_interactions(self, 
                data_path: str, 
                decider_col: str, 
                other_col: str, 
                like_col: str,
                timestamp_col: str
                ) -> None: 
    
    """
    Loads data and builds the interaction matrix.
    """
    try:
        # Load the data (validation happens inside DataLoader)
        self.data_loader.load_interactions(data_path, decider_col, other_col, like_col, timestamp_col)
        
        # Populate class attributes after loading
        self.interactions_graph = self.data_loader.interactions_graph
        self.interactions_df = self.data_loader.interactions_df
        self.user_df = self.data_loader.user_df
        
        print("Data Loaded ✅")
        
    except ValueError:
        print(f"❌ Error loading data")
        print("Please ensure your CSV has the required columns with correct data types:")
        print(f"- {decider_col}: numeric (user ID)")
        print(f"- {other_col}: numeric (user ID)")  
        print(f"- {like_col}: numeric (0/1 values)")
        print(f"- {timestamp_col}: valid datetime format")
        return
  
  def run_popularity(self) -> pd.DataFrame:
    """
    Computes popularity-based features for users.
    
    Returns:
        pd.DataFrame: DataFrame containing user features based on popularity models.
    """
    decider_col = self.data_loader.metadata['decider_col']
    other_col = self.data_loader.metadata['other_col']
    like_col = self.data_loader.metadata['like_col']

    # Calculate PageRank as a measure of popularity
    pagerank_scores = get_pagerank(self.interactions_graph)
    self.user_df = self.user_df.merge(pagerank_scores, on='user_id', how='left')

    # Calculate Like Stats
    like_stats = get_like_stats(self.interactions_df, decider_col, other_col, like_col)
    self.user_df = self.user_df.merge(like_stats, on='user_id', how='left')

    print("Popularity features added to User DF ✅")
  
  def is_ready(self) -> bool:
    """Check if data is loaded and ready for matching."""
    return self.data_loader.is_loaded()
  
  def get_user_interactions(self, user_id: int, as_decider: bool = True):
    """Get all interactions for a specific user."""
    return self.data_loader.get_user_interactions(user_id, as_decider)
