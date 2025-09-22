import pandas as pd
from .data.loader import DataLoader

class MatchingEngine:
  """High level API for performing matching tasks."""
  
  def __init__(self):
    self.data_loader = DataLoader()

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
    self.data_loader.load_interactions(data_path, decider_col, other_col, like_col, timestamp_col)
  
  @property
  def interactions_df(self):
    """Get the processed interactions DataFrame."""
    return self.data_loader.interactions_df
  
  @property 
  def interactions_graph(self):
    """Get the interaction graph."""
    return self.data_loader.interactions_graph
  
  @property
  def user_df(self):
    """Get the users DataFrame."""
    return self.data_loader.user_df
  
  def is_ready(self) -> bool:
    """Check if data is loaded and ready for matching."""
    return self.data_loader.is_loaded()
  
  def get_user_interactions(self, user_id: int, as_decider: bool = True):
    """Get all interactions for a specific user."""
    return self.data_loader.get_user_interactions(user_id, as_decider)
