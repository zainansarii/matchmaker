import pandas as pdimport pandas as pd

from .data.loader import DataLoaderfrom .data.loader import DataLo  import pandas as pd

from .models.graph import InteractionGraphfrom .data.loader import DataLoader

from .models.graph import InteractionGraph

class MatchingEngine:

  """High level API for performing matching tasks."""class MatchingEngine:

    """High level API for performing matching tasks."""

  def __init__(self):  

    self.data_loader = DataLoader()  def __init__(self):

    self.interaction_graph = InteractionGraph()    self.data_loader = DataLoader()

    self.interactions_df = None    self.interaction_graph = InteractionGraph()

    self.user_df = None    self.interactions_df = None

    self.user_df = None

  def load_interactions(self, 

                data_path: str,   def load_interactions(self, 

                decider_col: str,                 data_path: str, 

                other_col: str,                 decider_col: str, 

                like_col: str,                other_col: str, 

                timestamp_col: str                like_col: str,

                ) -> None:                 timestamp_col: str

                    ) -> None: 

    """    

    Loads data and builds the interaction matrix.    """

    """    Loads data and builds the interaction matrix.

    try:    """

        # Load the data (validation happens inside DataLoader)    try:

        self.data_loader.load_interactions(data_path, decider_col, other_col, like_col, timestamp_col)        # Load the data (validation happens inside DataLoader)

                self.data_loader.load_interactions(data_path, decider_col, other_col, like_col, timestamp_col)

        # Populate class attributes after loading        

        self.interactions_df = self.data_loader.interactions_df        # Populate class attributes after loading

        self.user_df = self.data_loader.user_df        self.interactions_df = self.data_loader.interactions_df

                self.user_df = self.data_loader.user_df

        # Build the interaction graph        

        self.interaction_graph.build_graph(        # Build the interaction graph

            self.interactions_df,         self.interaction_graph.build_graph(

            decider_col,             self.interactions_df, 

            other_col,             decider_col, 

            like_col            other_col, 

        )            like_col

                )

        print("Data Loaded ✅")        

                print("Data Loaded ✅")

    except ValueError:        

        # ValueError is raised by our validation - just show the custom message without traceback    except ValueError:

        print(f"❌ Error loading data")        # ValueError is raised by our validation - just show the custom message without traceback

        print("Please ensure your CSV has the required columns with correct data types:")        print(f"❌ Error loading data")

        print(f"- {decider_col}: numeric (user ID)")        print("Please ensure your CSV has the required columns with correct data types:")

        print(f"- {other_col}: numeric (user ID)")          print(f"- {decider_col}: numeric (user ID)")

        print(f"- {like_col}: numeric (0/1 values)")        print(f"- {other_col}: numeric (user ID)")  

        print(f"- {timestamp_col}: valid datetime format")        print(f"- {like_col}: numeric (0/1 values)")

        return        print(f"- {timestamp_col}: valid datetime format")

    except Exception as e:        return

        print(f"❌ Error loading data: {str(e)}")    except Exception as e:

        return        print(f"❌ Error loading data: {str(e)}")

          return

  def run_popularity(self) -> pd.DataFrame:  

    """  def run_popularity(self) -> pd.DataFrame:

    Computes popularity-based features for users.    """

        Computes popularity-based features for users.

    Returns:    

        pd.DataFrame: DataFrame containing user features based on popularity models.    Returns:

    """        pd.DataFrame: DataFrame containing user features based on popularity models.

    # Calculate PageRank as a measure of popularity    """

    pagerank_scores = self.interaction_graph.get_pagerank()    # Calculate PageRank as a measure of popularity

    self.user_df = self.user_df.merge(pagerank_scores, on='user_id', how='left')    pagerank_scores = self.interaction_graph.get_pagerank()

    self.user_df = self.user_df.merge(pagerank_scores, on='user_id', how='left')

    # Calculate Like Stats

    like_stats = self.interaction_graph.get_like_stats()    # Calculate Like Stats

    self.user_df = self.user_df.merge(like_stats, on='user_id', how='left')    like_stats = self.interaction_graph.get_like_stats()

    self.user_df = self.user_df.merge(like_stats, on='user_id', how='left')

    return self.user_df

      return self.user_df

  def is_ready(self) -> bool:  

    """Check if data is loaded and ready for matching."""  def is_ready(self) -> bool:

    return self.data_loader.is_loaded()    """Check if data is loaded and ready for matching."""

      return self.data_loader.is_loaded()

  def get_user_interactions(self, user_id: int, as_decider: bool = True):  

    """Get all interactions for a specific user."""  def get_user_interactions(self, user_id: int, as_decider: bool = True):

    return self.data_loader.get_user_interactions(user_id, as_decider)    """Get all interactions for a specific user."""
    return self.data_loader.get_user_interactions(user_id, as_decider)aph import InteractionGraph

class MatchingEngine:
  """High level API for performing matching tasks."""
  
  def __init__(self):
    self.data_loader = DataLoader()
    self.interaction_graph = InteractionGraph()
    self.interactions_df = None
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
        self.interactions_df = self.data_loader.interactions_df
        self.user_df = self.data_loader.user_df
        
        # Build the interaction graph
        self.interaction_graph.build_graph(
            self.interactions_df, 
            decider_col, 
            other_col, 
            like_col
        )
        
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
