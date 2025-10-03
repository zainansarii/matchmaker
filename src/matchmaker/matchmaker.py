import pandas as pd  # type: ignore[import]
from typing import Optional

from .data.loader import DataLoader
from .models.popularity import InteractionGraph, get_like_stats, assign_balanced_leagues
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
                          timestamp_col: str,
                          gender_col: Optional[str] = None
                          ) -> None:
        """
        Loads data into interaction matrix, interaction graph, and fits ALS
        
        Args:
            data_path: Path to CSV file containing interaction data
            decider_col: Column name for the user making the decision
            other_col: Column name for the target user
            like_col: Column name for the interaction type (like/dislike)  
            timestamp_col: Column name for the timestamp
            gender_col: Column name for the decider's gender (will infer other genders assuming heterosexual interactions)
        """
        try:
            print("Reading data... ", end="")
            # Load the data (validation happens inside DataLoader)
            self.data_loader.load_interactions(data_path, decider_col, other_col, like_col, timestamp_col, gender_col)
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
                self.user_df,
                decider_col,
                other_col,
                like_col,
                gender_col="gender"
            )         
            print("Complete! ✅")

        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            import traceback
            traceback.print_exc()
            return

    def run_popularity(self):
        """
        Computes popularity-based features for users.
        
        Returns:
            pd.DataFrame: DataFrame containing user features based on popularity models.
        """
        # Remove existing popularity columns to avoid conflicts
        popularity_columns = ['pagerank', 'likes_received', 'swipes_received', 'weighted_likes_received', 
                            'weighted_swipes_received', 'in_like_rate_raw', 'in_like_rate_weighted',
                            'in_like_rate_smoothed', 'popularity_confidence', 'popularity_score']
        
        existing_columns = list(self.user_df.columns)
        columns_to_drop = [col for col in existing_columns if any(pop_col in col for pop_col in popularity_columns)]
        
        if columns_to_drop:
            self.user_df = self.user_df.drop(columns=columns_to_drop)
        
        # Calculate PageRank as a measure of popularity (from graph class)
        pagerank_scores = self.interaction_graph.get_pagerank()
        self.user_df = self.user_df.merge(pagerank_scores, on='user_id', how='left')

        # Calculate in-like rate statistics (standalone function)
        decider_col = self.data_loader.metadata['decider_col']
        other_col = self.data_loader.metadata['other_col']
        like_col = self.data_loader.metadata['like_col']
        timestamp_col = self.data_loader.metadata['timestamp_col']
        
        like_stats = get_like_stats(
            self.interaction_df, 
            decider_col, 
            other_col, 
            like_col, 
            timestamp_col=timestamp_col,
            recency_halflife_days=30.0,
            min_swipes=3
        )
        self.user_df = self.user_df.merge(like_stats, on='user_id', how='left')

        # Assign balanced leagues by gender using PageRank
        try:
            leagues = assign_balanced_leagues(self.user_df, pagerank_col='pagerank', gender_col='gender')
            self.user_df = self.user_df.merge(leagues, on='user_id', how='left')
        except Exception as e:
            # Don't fail the pipeline if leagues can't be assigned; just log
            print(f"⚠️ League assignment skipped: {e}")

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
