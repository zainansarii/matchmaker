import pandas as pd  # type: ignore[import]

from .data.loader import DataLoader
from .models.graph import InteractionGraph
from .models.als import ALSModel
from .models.engagement import EngagementScorer, EngagementConfig


class MatchingEngine:
    """High level API for performing matching tasks."""

    def __init__(self):
        self.data_loader = DataLoader()
        self.interaction_graph = InteractionGraph()
        self.als_model = ALSModel()
        self.engagement_model = EngagementScorer()
        self.interaction_df = None
        self.user_df = None

    def load_interactions(self, 
                          data_path: str, 
                          decider_col: str, 
                          other_col: str, 
                          like_col: str,
                          timestamp_col: str
                          ) -> None:
        """
        Loads data into interaction matrix, interaction graph, and fits ALS
        """
        try:
            print("Reading data... ", end="")
            # Load the data (validation happens inside DataLoader)
            self.data_loader.load_interactions(data_path, decider_col, other_col, like_col, timestamp_col)
            # Populate class attributes after loading
            self.interaction_df = self.data_loader.interaction_df
            self.user_df = self.data_loader.user_df
            print("✅")

            print("Constructing graph...", end="")
            self.interaction_graph.build_graph(
                self.interaction_df, 
                decider_col, 
                other_col, 
                like_col
            )
            print("✅")

            print("Fitting ALS... ", end="")
            self.als_model.fit(
                self.interaction_df, 
                decider_col, 
                other_col, 
                like_col
            )            
            print("Complete! ✅")
            
        except ValueError:
            # ValueError is raised by our validation - just show the custom message without traceback
            print(f"❌ Error loading data")
            print("Please ensure your CSV has the required columns with correct data types:")
            print(f"- {decider_col}: numeric (user ID)")
            print(f"- {other_col}: numeric (user ID)")
            print(f"- {like_col}: numeric (0/1 values)")
            print(f"- {timestamp_col}: valid datetime format")
            return
        
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return

    def run_popularity(self) -> pd.DataFrame:
        """
        Computes popularity-based features for users.
        
        Returns:
            pd.DataFrame: DataFrame containing user features based on popularity models.
        """
        # Calculate PageRank as a measure of popularity
        pagerank_scores = self.interaction_graph.get_pagerank()
        self.user_df = self.user_df.merge(pagerank_scores, on='user_id', how='left')

        print("User DF updated ✅")

    def run_engagement(self, config: EngagementConfig | None = None) -> pd.DataFrame:
        """Compute engagement scores and merge them into the user feature table."""

        if not self.is_ready():
            raise ValueError("Data not loaded. Call load_interactions() first.")

        metadata = self.data_loader.metadata
        decider_col = metadata['decider_col']
        like_col = metadata['like_col']
        timestamp_col = metadata.get('timestamp_col')

        if config is not None:
            self.engagement_model = EngagementScorer(config)

        engagement_scores = self.engagement_model.score(
            self.interaction_df,
            decider_col=decider_col,
            like_col=like_col,
            timestamp_col=timestamp_col,
        )

        self.user_df = self.user_df.merge(engagement_scores, on='user_id', how='left')

        print("User DF updated ✅")

    def is_ready(self) -> bool:
        """Check if data is loaded and ready for matching."""
        return self.data_loader.is_loaded()

    def get_user_interactions(self, user_id: int, as_decider: bool = True):
        """Get all interactions for a specific user."""
        return self.data_loader.get_user_interactions(user_id, as_decider)
