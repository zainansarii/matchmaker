"""
Test the full recommendation pipeline with optimizations
"""
import time
import sys
sys.path.insert(0, '/home/zain/projects/matchmaker/src')

from matchmaker import matchmaker
import cudf

print("Loading engine...")
engine = matchmaker.MatchingEngine()
engine.load_interactions("/home/zain/projects/matchmaker/examples/data/swipes_clean.csv", 
  decider_col='decidermemberid',
  other_col='othermemberid', 
  like_col='like', 
  timestamp_col='timestamp',
  gender_col='decidergender')
engine.run_engagement()
engine.run_popularity()
engine.build_recommender()

# Get test users
user_df = engine.user_df
print("\n" + "="*60)
print("Testing batch recommendation performance")
print("="*60)

for n_users in [10, 100, 1000]:
    test_users = user_df[user_df.gender=='M'].dropna().sample(n_users).user_id.to_arrow().to_pylist()
    
    start = time.time()
    batch_results = engine.recommend_batch(test_users, k=1000)
    elapsed = time.time() - start
    
    print(f"\n{n_users:4d} users: {elapsed:.2f}s total, {elapsed*1000/n_users:.1f}ms per user")
    print(f"        Sample result: {len(batch_results[test_users[0]])} recommendations")

print("\n" + "="*60)
print("✅ Optimization complete!")
