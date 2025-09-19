import pandas as pd
from .data import loader

class MatchingEngine:
  """High level API for performing matching tasks."""
  
  def __init__(self):
    pass

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

    self.interactions_df, self.interactions_graph = loader.load_interactions(data_path, decider_col, other_col, like_col, timestamp_col)
