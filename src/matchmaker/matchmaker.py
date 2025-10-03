import pandas as pd  # type: ignore[import]
from typing import Optional, List, Tuple, Dict

from .data.loader import DataLoader
from .models.popularity import InteractionGraph, get_like_stats, assign_balanced_leagues
from .models.als import ALSModel
from .models.engagement import EngagementScorer, EngagementConfig
from .models.recommender import LeagueFilteredRecommender


class MatchingEngine:
    """High level API for performing matching tasks."""

    def __init__(self):
        self.data_loader = DataLoader()
        self.interaction_graph = InteractionGraph()
        self.als_model = ALSModel()
        self.engagement_model = EngagementScorer()
        self.recommender = None  # Initialized after run_popularity()
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
        # Only assign leagues to users who are in the ALS model (have sufficient interactions)
        try:
            # Get ALL users that appear in ANY ALS model
            male_ids_in_als = set(self.als_model.male_map.keys()) | set(self.als_model.male_map_f2m.keys())
            female_ids_in_als = set(self.als_model.female_map.keys()) | set(self.als_model.female_map_f2m.keys())
            als_user_ids = male_ids_in_als | female_ids_in_als
            
            # Filter to only ALS users for league assignment
            als_users_df = self.user_df[self.user_df['user_id'].isin(als_user_ids)]
            
            if len(als_users_df) > 0:
                leagues = assign_balanced_leagues(als_users_df, pagerank_col='pagerank', gender_col='gender')
                self.user_df = self.user_df.merge(leagues[['user_id', 'league']], on='user_id', how='left')
            else:
                print("⚠️ No users in ALS model for league assignment")
        except Exception as e:
            # Don't fail the pipeline if leagues can't be assigned; just log
            print(f"⚠️ League assignment skipped: {e}")

        print("User DF updated ✅")
    
    def build_recommender(self, use_gpu: bool = True):
        """
        Build FAISS-based league-filtered recommender.
        
        Must be called after run_popularity() to ensure leagues are assigned.
        
        Args:
            use_gpu: Whether to use GPU for FAISS indices (default True)
        """
        if not self.is_ready():
            raise ValueError("Data not loaded. Call load_interactions() first.")
        
        if 'league' not in self.user_df.columns:
            raise ValueError("Leagues not assigned. Call run_popularity() first.")
        
        try:
            print("Building FAISS recommender... ", end="")
            self.recommender = LeagueFilteredRecommender(
                als_model=self.als_model,
                user_df=self.user_df,
                use_gpu=use_gpu
            )
            print("✅")
        except Exception as e:
            print(f"⚠️ FAISS recommender initialization failed: {e}")
            self.recommender = None
            raise

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
    
    def recommend_for_user(self, user_id: int, k: int = 1000) -> List[Tuple[int, float]]:
        """
        Get top-k recommendations for a user, filtered by league and ranked by mutual score.
        
        Args:
            user_id: User ID to generate recommendations for
            k: Number of recommendations to return (default 1000)
            
        Returns:
            List of (candidate_id, mutual_score) tuples, sorted by score descending
        """
        if self.recommender is None:
            raise ValueError("Recommender not initialized. Call run_popularity() first.")
        
        # Get user metadata
        user_row = self.user_df[self.user_df['user_id'] == user_id]
        if len(user_row) == 0:
            return []
        
        gender = user_row['gender'].iloc[0]
        league = user_row['league'].iloc[0]
        
        if gender == 'M':
            return self.recommender.recommend_for_male(user_id, league, k)
        elif gender == 'F':
            return self.recommender.recommend_for_female(user_id, league, k)
        else:
            return []
    
    def recommend_batch(self, user_ids: List[int], k: int = 1000) -> Dict[int, List[Tuple[int, float]]]:
        """
        Get top-k recommendations for multiple users in batch.
        
        Args:
            user_ids: List of user IDs
            k: Number of recommendations per user
            
        Returns:
            Dict mapping user_id -> [(candidate_id, mutual_score), ...]
        """
        if self.recommender is None:
            raise ValueError("Recommender not initialized. Call run_popularity() first.")
        
        # Build metadata dict
        user_metadata = {}
        for uid in user_ids:
            user_row = self.user_df[self.user_df['user_id'] == uid]
            if len(user_row) > 0:
                user_metadata[uid] = {
                    'gender': user_row['gender'].iloc[0],
                    'league': user_row['league'].iloc[0]
                }
        
        return self.recommender.recommend_batch(user_ids, user_metadata, k)
