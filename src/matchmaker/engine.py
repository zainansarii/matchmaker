import cudf
import pandas as pd  # type: ignore[import]
from typing import Optional, List, Tuple, Dict

from .data.loader import DataLoader
from .models.als import ALSModel
from .models.engagement import EngagementScorer, EngagementConfig
from .models.elo import EloRatingSystem, EloConfig, assign_leagues_from_elo
from .serving import LeagueFilteredRecommender, ALSFaissRecommender


class MatchingEngine:
    """High level API for performing matching tasks."""

    def __init__(self):
        self.data_loader = DataLoader()
        self.als_model = ALSModel()
        self.engagement_model = EngagementScorer()
        self.elo_model = EloRatingSystem()
        self.recommender = None  # Initialized after leagues are assigned
        self.recommender_variant: Optional[str] = None
        self.interaction_df = None
        self.user_df = None

    def load_interactions(
        self,
        data_path: str,
        decider_col: str,
        other_col: str,
        like_col: str,
        timestamp_col: str,
        gender_col: Optional[str] = None,
    ) -> None:
        """Load interactions into memory and fit the ALS model."""
        try:
            print("Reading data... ", end="")
            # Load the data (validation happens inside DataLoader)
            self.data_loader.load_interactions(data_path, decider_col, other_col, like_col, timestamp_col, gender_col)
            # Populate class attributes after loading
            self.interaction_df = self.data_loader.interaction_df
            self.user_df = self.data_loader.user_df
            print("✅")

            print("Fitting ALS... ")
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

    def build_recommender(self, variant: str = "pop", use_gpu: bool = True):
        """
        Build a FAISS-based recommender.

        Args:
            variant: Which serving strategy to use. Supported values:
                - "pop": league-filtered recommender that enforces attractiveness tiers.
                - "vanilla": ALS + FAISS without league constraints.
            use_gpu: Must remain True; CPU execution is no longer supported.

        Notes:
            The "pop" variant requires leagues computed via ``run_elo``.
        """
        if not self.is_ready():
            raise ValueError("Data not loaded. Call load_interactions() first.")

        if not use_gpu:
            raise ValueError("GPU is required for FAISS recommenders; CPU fallback has been removed.")
        
        normalized_variant = variant.lower()
        if normalized_variant not in {"pop", "vanilla"}:
            raise ValueError("variant must be either 'pop' or 'vanilla'")

        try:
            if normalized_variant == "pop":
                if 'league' not in self.user_df.columns:
                    raise ValueError("Leagues not assigned. Call run_elo() first to use the 'pop' recommender.")

                print("Building FAISS recommender (pop)... ", end="")
                self.recommender = LeagueFilteredRecommender(
                    als_model=self.als_model,
                    user_df=self.user_df,
                    use_gpu=use_gpu
                )
            else:
                print("Building FAISS recommender (vanilla)... ", end="")
                self.recommender = ALSFaissRecommender(
                    als_model=self.als_model,
                    use_gpu=use_gpu
                )

            self.recommender_variant = normalized_variant
            print("✅")
        except Exception as e:
            print(f"⚠️ FAISS recommender initialization failed: {e}")
            self.recommender = None
            self.recommender_variant = None
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
    
    def run_elo(self, config: EloConfig | None = None) -> pd.DataFrame:
        """
        Compute ELO ratings and assign leagues based on gender-specific percentiles.

        Args:
            config: ELO configuration (uses defaults if None)

        Returns:
            DataFrame with ELO ratings merged into user_df
        """
        if not self.is_ready():
            raise ValueError("Data not loaded. Call load_interactions() first.")

        metadata = self.data_loader.metadata
        decider_col = metadata['decider_col']
        other_col = metadata['other_col']
        like_col = metadata['like_col']
        timestamp_col = metadata.get('timestamp_col')
        gender_col = metadata.get('gender_col')

        if config is not None:
            self.elo_model = EloRatingSystem(config)
        
        # If gender_col is specified, we need to join gender info into interactions
        interactions_with_gender = self.interaction_df
        if gender_col is not None:
            # Join gender from user_df to interaction_df
            gender_lookup = self.user_df[['user_id', 'gender']].rename(
                columns={'user_id': decider_col, 'gender': gender_col}
            )
            interactions_with_gender = self.interaction_df.merge(
                gender_lookup, on=decider_col, how='left'
            )
        
        elo_scores = self.elo_model.score(
            interactions_with_gender,
            decider_col=decider_col,
            other_col=other_col,
            like_col=like_col,
            timestamp_col=timestamp_col,
            gender_col=gender_col
        )
        
        # Remove existing ELO columns if they exist
        elo_columns = ['elo_rating', 'interaction_count', 'is_stable']
        existing_cols = list(self.user_df.columns)
        cols_to_drop = [col for col in existing_cols if col in elo_columns]
        if cols_to_drop:
            self.user_df = self.user_df.drop(columns=cols_to_drop)
        
        # Drop gender from elo_scores if it exists (user_df already has it)
        if 'gender' in elo_scores.columns:
            elo_scores = elo_scores.drop(columns=['gender'])

        # Merge ELO scores
        self.user_df = self.user_df.merge(elo_scores, on='user_id', how='left')

        # Assign leagues based on updated ELO ratings
        if 'league' in self.user_df.columns:
            self.user_df = self.user_df.drop(columns=['league'])

        league_assignments = assign_leagues_from_elo(self.user_df, self.als_model)
        self.user_df = self.user_df.merge(league_assignments, on='user_id', how='left')

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
        if self.recommender is None or self.recommender_variant is None:
            raise ValueError("Recommender not initialized. Call build_recommender() first.")
        
        # Get user metadata
        user_row = self.user_df[self.user_df['user_id'] == user_id]
        if len(user_row) == 0:
            return []
        
        gender = user_row['gender'].iloc[0]
        league = user_row['league'].iloc[0] if 'league' in user_row.columns else None

        if self.recommender_variant == "pop":
            if league is None:
                raise ValueError("League information missing for user; cannot use 'pop' recommender.")
            if gender == 'M':
                return self.recommender.recommend_for_male(user_id, league, k)
            if gender == 'F':
                return self.recommender.recommend_for_female(user_id, league, k)
            return []

        # Vanilla fallback (no league filtering)
        if gender == 'M':
            return self.recommender.recommend_for_male(user_id, k)
        if gender == 'F':
            return self.recommender.recommend_for_female(user_id, k)
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
        if self.recommender is None or self.recommender_variant is None:
            raise ValueError("Recommender not initialized. Call build_recommender() first.")
        
        # Build metadata dict with efficient single DataFrame operation
        user_ids_set = set(user_ids)
        
        columns = ['user_id', 'gender']
        has_league = 'league' in self.user_df.columns
        if has_league:
            columns.append('league')

        # Convert to pandas if needed for faster iteration
        if isinstance(self.user_df, cudf.DataFrame):
            users_subset = self.user_df[self.user_df['user_id'].isin(user_ids_set)][columns].to_pandas()
        else:
            users_subset = self.user_df[self.user_df['user_id'].isin(user_ids_set)][columns]
        
        # Build metadata dict from the subset
        user_metadata: Dict[int, Dict[str, Optional[str]]] = {}
        for _, row in users_subset.iterrows():
            meta = {'gender': row['gender']}
            if has_league:
                meta['league'] = row['league']
            user_metadata[row['user_id']] = meta
        
        return self.recommender.recommend_batch(user_ids, user_metadata, k)

