import numpy as np
import pandas as pd
import cupy as cp
from scipy import sparse as sp
from implicit.als import AlternatingLeastSquares
import warnings
warnings.filterwarnings('ignore')

class ALS:
    """
    GPU-accelerated collaborative filtering for dating app recommendations.
    Uses Implicit ALS with square user-user matrix structure.
    """
    
    def __init__(self, 
                 factors=64,           
                 regularization=0.01,  
                 iterations=15,        
                 alpha=40.0,          
                 use_gpu=True,
                 random_state=42):
        """
        Initialize the collaborative filtering model.
        
        Args:
            factors: Number of latent factors
            regularization: L2 regularization strength
            iterations: Number of ALS iterations
            alpha: Confidence weighting for implicit feedback
            use_gpu: Whether to use GPU acceleration
            random_state: Random seed for reproducibility
        """
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.random_state = random_state
        
        # Check GPU availability
        self.use_gpu = use_gpu and self._check_gpu_available()
        
        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            use_gpu=self.use_gpu,
            random_state=random_state,
            dtype=np.float32
        )
        
        # Mappings and matrices
        self.user2idx = None
        self.idx2user = None
        self.user_user_matrix = None
        
        # Cache for performance
        self.user_factors = None
        self.item_factors = None
        
        # Stats
        self.n_users = 0
        self.n_interactions = 0
        
    def _check_gpu_available(self):
        """Check if GPU is available for computation"""
        try:
            test = cp.array([1, 2, 3])
            del test
            print("✅ GPU acceleration enabled")
            return True
        except:
            print("⚠️ GPU not available, falling back to CPU")
            return False
    
    def _convert_factors_to_numpy(self, factors):
        """Convert implicit factors to numpy arrays, handling both CPU and GPU matrices"""
        try:
            # Method 1: Use .to_numpy() if available (for implicit GPU matrices)
            if hasattr(factors, 'to_numpy'):
                return factors.to_numpy()
            # Method 2: Use cupy.asnumpy() for CuPy arrays
            elif hasattr(factors, 'get'):
                return factors.get()
            # Method 3: Direct numpy conversion
            else:
                return np.array(factors)
        except Exception as e:
            print(f"Warning: Factor conversion failed with {type(factors)}: {e}")
            # Fallback - return the original factors (may cause issues)
            return factors
    
    def fit(self, interactions, 
            user_col='decidermemberid', 
            target_col='othermemberid', 
            value_col='like',
            min_interactions=2):
        """
        Fit the collaborative filtering model on interaction data.
        
        Args:
            interactions: DataFrame with user interactions (can be pandas or cudf)
            user_col: Column name for users making decisions
            target_col: Column name for target users
            value_col: Column name for interaction values (0/1)
            min_interactions: Minimum interactions required per user
        """
        print("🚀 Starting collaborative filtering training...")
        total_start = pd.Timestamp.now()
        
        # Handle both pandas and cudf DataFrames
        if hasattr(interactions, 'to_pandas'):
            # It's a cuDF DataFrame
            print("📊 Detected cuDF DataFrame, converting to pandas...")
            df = interactions[[user_col, target_col, value_col]].to_pandas()
        else:
            # It's a pandas DataFrame
            df = interactions[[user_col, target_col, value_col]].copy()
        
        # Filter out users with too few interactions
        user_counts = df.groupby(user_col)[value_col].count().reset_index()
        user_counts.columns = [user_col, 'count']
        active_users = user_counts[user_counts['count'] >= min_interactions][user_col]
        
        # Filter interactions
        df = df[df[user_col].isin(active_users)]
        
        # Create user mappings - get all unique users from both columns
        print("📊 Creating user mappings...")
        all_users = set(df[user_col].unique()) | set(df[target_col].unique())
        # Filter to only include active users
        all_users = list(all_users.intersection(set(active_users)))
        
        self.n_users = len(all_users)
        
        # Create mappings
        self.user2idx = {user: idx for idx, user in enumerate(all_users)}
        self.idx2user = {idx: user for user, idx in self.user2idx.items()}
        
        print(f"📈 {self.n_users:,} users in square matrix")
        
        # Map to indices
        df['user_idx'] = df[user_col].map(self.user2idx)
        df['target_idx'] = df[target_col].map(self.user2idx)
        
        # Remove unmapped entries (users not in active list)
        df = df.dropna(subset=['user_idx', 'target_idx'])
        df['user_idx'] = df['user_idx'].astype(np.int32)
        df['target_idx'] = df['target_idx'].astype(np.int32)
        
        # Apply confidence weighting (implicit feedback)
        df['confidence'] = 1 + self.alpha * df[value_col].astype(np.float32)
        
        # Remove duplicates by taking max confidence
        df = df.groupby(['user_idx', 'target_idx'])['confidence'].max().reset_index()
        
        self.n_interactions = len(df)
        print(f"💾 {self.n_interactions:,} unique interactions")
        
        # Build square sparse matrix (user x user)
        print("🔨 Building square sparse matrix...")
        
        self.user_user_matrix = sp.csr_matrix(
            (df['confidence'].values, 
             (df['user_idx'].values, df['target_idx'].values)),
            shape=(self.n_users, self.n_users),
            dtype=np.float32
        )
        
        density = 100.0 * self.n_interactions / (self.n_users * self.n_users)
        print(f"📊 Matrix density: {density:.4f}%")
        
        # Fit the model - for ALS, we need to transpose for item-user format
        print("🎯 Training ALS model...")
        fit_start = pd.Timestamp.now()
        
        self.model.fit(self.user_user_matrix.T.tocsr(), show_progress=True)
        
        fit_time = (pd.Timestamp.now() - fit_start).total_seconds()
        total_time = (pd.Timestamp.now() - total_start).total_seconds()
        
        # Cache factors for faster recommendations - convert to numpy properly
        print("🔄 Converting factors to numpy arrays...")
        self.user_factors = self._convert_factors_to_numpy(self.model.user_factors)
        self.item_factors = self._convert_factors_to_numpy(self.model.item_factors)
        
        print(f"✅ Model trained in {fit_time:.2f}s")
        print(f"⏱️ Total processing time: {total_time:.2f}s")
        print(f"📋 Factor arrays: user_factors {self.user_factors.shape}, item_factors {self.item_factors.shape}")
        
        return self
    
    def recommend_for_user(self, user_id, N=10, filter_already_liked=True, return_scores=True):
        """
        Get top N recommendations for a specific user.
        
        Args:
            user_id: User ID to get recommendations for
            N: Number of recommendations
            filter_already_liked: Whether to filter out already liked users
            return_scores: Whether to return scores with recommendations
        
        Returns:
            List of (user_id, score) tuples or just user_ids
        """
        if user_id not in self.user2idx:
            print(f"⚠️ User {user_id} not found in training data")
            return []
        
        user_idx = self.user2idx[user_id]
        
        # Extract the user's row from the matrix
        user_items = self.user_user_matrix[user_idx]
        
        # Get recommendations using correct method signature
        try:
            recommendations, scores = self.model.recommend(
                user_idx,
                user_items,
                N=N,
                recalculate_user=False
            )
        except TypeError:
            # Fallback for different implicit versions
            recommendations, scores = self.model.recommend(
                user_idx,
                user_items,
                N=N
            )
        
        # Manual filtering of already liked items if requested
        if filter_already_liked:
            already_liked = set(user_items.indices[user_items.data > 1])
            
            filtered_results = []
            for target_idx, score in zip(recommendations, scores):
                if int(target_idx) not in already_liked:
                    filtered_results.append((target_idx, score))
            
            # Take only the top N after filtering
            filtered_results = filtered_results[:N]
            recommendations = [r[0] for r in filtered_results]
            scores = [r[1] for r in filtered_results]
        
        # Map back to user IDs
        results = []
        for target_idx, score in zip(recommendations, scores):
            target_id = self.idx2user.get(int(target_idx))
            if target_id is not None:
                if return_scores:
                    results.append((target_id, float(score)))
                else:
                    results.append(target_id)
        
        return results
    
    def recommend_batch(self, user_ids, N=10, filter_already_liked=True, batch_size=1000):
        """
        Ultra-optimized batch recommendations using matrix operations with memory management.
        
        This version processes users in batches to avoid memory issues while still
        being much faster than individual processing.
        
        Args:
            user_ids: List of user IDs
            N: Number of recommendations per user
            filter_already_liked: Whether to filter out already liked users
            batch_size: Number of users to process at once (controls memory usage)
        
        Returns:
            Dictionary mapping user_id to list of recommendations
        """
        results = {}
        
        # Filter valid users
        valid_users = [uid for uid in user_ids if uid in self.user2idx]
        if not valid_users:
            return results
        
        print(f"🚀 Processing {len(valid_users)} users in batches of {batch_size}")
        
        # Process in batches
        for batch_start in tqdm(range(0, len(valid_users), batch_size)):
            batch_end = min(batch_start + batch_size, len(valid_users))
            batch_users = valid_users[batch_start:batch_end]
            
            try:
                # Convert to indices for this batch
                user_indices = np.array([self.user2idx[uid] for uid in batch_users])
                
                # Get user factors for this batch
                user_factors_batch = self.user_factors[user_indices]  # Shape: (batch_size, n_factors)
                
                # Compute scores for all users in batch against all items
                batch_scores = np.dot(user_factors_batch, self.item_factors.T)  # Shape: (batch_size, n_users)
                
                # Process each user's scores in this batch
                for i, uid in enumerate(batch_users):
                    user_idx = user_indices[i]
                    user_scores = batch_scores[i].copy()  # Copy to avoid modifying original
                    
                    # Apply filtering if requested
                    if filter_already_liked:
                        user_items = self.user_user_matrix[user_idx]
                        already_liked = user_items.indices[user_items.data > 1]
                        user_scores[already_liked] = -np.inf  # Mask out already liked items
                    
                    # Also mask out the user themselves
                    user_scores[user_idx] = -np.inf
                    
                    # Get top N recommendations efficiently
                    if len(user_scores) > N:
                        # Use argpartition for better performance when N << total_users
                        top_indices = np.argpartition(user_scores, -N)[-N:]
                        top_indices = top_indices[np.argsort(user_scores[top_indices])][::-1]
                    else:
                        # If we have fewer items than N, just sort all
                        top_indices = np.argsort(user_scores)[::-1]
                    
                    # Convert to user IDs and scores
                    user_results = []
                    for idx in top_indices:
                        if user_scores[idx] > -np.inf:  # Valid recommendation
                            target_id = self.idx2user.get(int(idx))
                            if target_id is not None:
                                user_results.append((target_id, float(user_scores[idx])))
                        if len(user_results) >= N:  # Stop once we have enough
                            break
                    
                    results[uid] = user_results
                    
            except Exception as e:
                print(f"⚠️ Batch processing failed for batch {batch_start//batch_size + 1} ({e})")
                print("🔄 Falling back to individual processing for this batch...")
                
                # Fallback to individual processing for this batch only
                for uid in batch_users:
                    try:
                        results[uid] = self.recommend_for_user(uid, N, filter_already_liked)
                    except Exception as individual_error:
                        print(f"⚠️ Individual processing failed for user {uid}: {individual_error}")
                        results[uid] = []
        
        print(f"✅ Completed batch processing for {len(results)} users")
        return results
    
    def find_similar_users(self, user_id, N=10):
        """
        Find users with similar preferences.
        
        Args:
            user_id: User ID to find similar users for
            N: Number of similar users to return
        
        Returns:
            List of (user_id, similarity_score) tuples
        """
        if user_id not in self.user2idx:
            return []
        
        user_idx = self.user2idx[user_id]
        
        # Use numpy arrays for consistent computation
        user_vector = self.user_factors[user_idx]
        all_factors = self.user_factors
        
        # Compute similarities with all other users - CPU version for consistency
        try:
            norms = np.linalg.norm(all_factors, axis=1)
            user_norm = np.linalg.norm(user_vector)
            similarities = np.dot(all_factors, user_vector) / (norms * user_norm + 1e-8)
        except Exception as e:
            print(f"Error computing similarities: {e}")
            return []
        
        # Get top N similar users (excluding self)
        similarities[user_idx] = -1  # Exclude self
        top_indices = np.argsort(similarities)[-N-1:-1][::-1]
        
        results = []
        for idx in top_indices:
            if idx != user_idx and similarities[idx] > 0:
                similar_user_id = self.idx2user.get(int(idx))
                if similar_user_id:
                    results.append((similar_user_id, float(similarities[idx])))
        
        return results[:N]
    
    def get_user_stats(self, user_id):
        """
        Get statistics for a specific user.
        
        Args:
            user_id: User ID to get stats for
        
        Returns:
            Dictionary with user statistics
        """
        if user_id not in self.user2idx:
            return None
        
        user_idx = self.user2idx[user_id]
        
        # Count likes given (row in matrix)
        user_row = self.user_user_matrix[user_idx]
        likes_given = (user_row.data > 1).sum()
        total_evaluated = user_row.nnz
        
        # Count likes received (column in matrix)
        user_col = self.user_user_matrix[:, user_idx]
        likes_received = (user_col.data > 1).sum()
        
        return {
            'user_id': user_id,
            'profiles_evaluated': int(total_evaluated),
            'likes_given': int(likes_given),
            'likes_received': int(likes_received),
            'like_rate': float(likes_given / max(total_evaluated, 1)),
            'selectivity': 1.0 - float(likes_given / max(total_evaluated, 1))
        }
    
    def explain_recommendation(self, user_id, recommended_user_id, N=5):
        """
        Explain why a user was recommended to another user.
        
        Args:
            user_id: User ID
            recommended_user_id: Recommended user ID
            N: Number of similar users to show as explanation
        
        Returns:
            Dictionary with explanation details
        """
        if user_id not in self.user2idx or recommended_user_id not in self.user2idx:
            return None
        
        user_idx = self.user2idx[user_id]
        rec_user_idx = self.user2idx[recommended_user_id]
        
        # Get users who liked the recommended user
        rec_user_col = self.user_user_matrix[:, rec_user_idx].tocoo()
        users_who_liked = [(self.idx2user[idx], float(data)) 
                          for idx, data in zip(rec_user_col.row, rec_user_col.data) 
                          if data > 1 and idx in self.idx2user]
        
        # Find similar users among those who liked
        similar_users = self.find_similar_users(user_id, N=50)
        similar_who_liked = []
        
        for similar_user, similarity in similar_users:
            if any(u[0] == similar_user for u in users_who_liked):
                similar_who_liked.append((similar_user, similarity))
                if len(similar_who_liked) >= N:
                    break
        
        # Calculate predicted score using numpy arrays
        user_vector = self.user_factors[user_idx]
        rec_user_vector = self.item_factors[rec_user_idx]
        predicted_score = float(np.dot(user_vector, rec_user_vector))
        
        return {
            'user_id': user_id,
            'recommended_user_id': recommended_user_id,
            'predicted_score': predicted_score,
            'total_likes': len(users_who_liked),
            'similar_users_who_liked': similar_who_liked,
            'explanation': f"Recommended because {len(similar_who_liked)} similar users also liked this profile"
        }
    
    def save_model(self, filepath):
        """Save the trained model to disk"""
        import pickle
        
        model_data = {
            'model_state': self.model,
            'user2idx': self.user2idx,
            'idx2user': self.idx2user,
            'n_users': self.n_users,
            'n_interactions': self.n_interactions,
            'user_factors': self.user_factors,
            'item_factors': self.item_factors
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model from disk"""
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model_state']
        self.user2idx = model_data['user2idx']
        self.idx2user = model_data['idx2user']
        self.n_users = model_data['n_users']
        self.n_interactions = model_data['n_interactions']
        self.user_factors = model_data['user_factors']
        self.item_factors = model_data['item_factors']
        
        print(f"✅ Model loaded from {filepath}")
        return self