#!/usr/bin/env python
"""Profile batch recommendation performance to find bottlenecks."""

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

print("\nRunning engagement...")
engine.run_engagement()

print("\nRunning popularity...")
engine.run_popularity()

print("\nBuilding recommender...")
engine.build_recommender()

# Get test users
user_df = engine.user_df
test_users = user_df[user_df.gender=='M'].dropna().sample(10).user_id.to_arrow().to_pylist()

# Check FAISS
import faiss
print(f"\n{'='*60}")
print(f"FAISS GPUs available: {faiss.get_num_gpus()}")
print(f"Engine recommender use_gpu: {engine.recommender.use_gpu}")
print(f"{'='*60}")

# Test single user
print("\n=== Single User Test ===")
start = time.time()
single_result = engine.recommend_for_user(test_users[0], k=1000)
single_time = time.time() - start
print(f"Time: {single_time*1000:.1f}ms")
print(f"Results: {len(single_result)}")

# Test batch (10 users)
print("\n=== Batch Test (10 users) ===")
start = time.time()
batch_results = engine.recommend_batch(test_users, k=1000)
batch_time = time.time() - start
print(f"Total time: {batch_time*1000:.1f}ms")
print(f"Per user: {batch_time*100:.1f}ms")
print(f"Sample results: {len(batch_results[test_users[0]])}")

# Test larger batch (100 users)
print("\n=== Batch Test (100 users) ===")
test_users_100 = user_df[user_df.gender=='M'].dropna().sample(100).user_id.to_arrow().to_pylist()
start = time.time()
batch_results_100 = engine.recommend_batch(test_users_100, k=1000)
batch_time_100 = time.time() - start
print(f"Total time: {batch_time_100:.2f}s")
print(f"Per user: {batch_time_100*10:.1f}ms")

print(f"\n{'='*60}")
print("Performance Summary:")
print(f"  Single user:  {single_time*1000:.1f}ms")
print(f"  10 users:     {batch_time*1000:.1f}ms ({batch_time*100:.1f}ms/user)")
print(f"  100 users:    {batch_time_100:.2f}s ({batch_time_100*10:.1f}ms/user)")
print(f"{'='*60}")


# Single user
print("\n=== Single User ===")
start = time.time()
result = engine.recommend_for_user(test_users[0], k=1000)
single_time = time.time() - start
print(f"{single_time*1000:.1f}ms, {len(result)} results")

# Batch 10
print("\n=== Batch 10 ===")
start = time.time()
results = engine.recommend_batch(test_users, k=1000)
print(f"{(time.time()-start)*1000:.1f}ms total, {(time.time()-start)*100:.1f}ms/user")

# Batch 100
print("\n=== Batch 100 ===")
test_100 = user_df[user_df.gender=='M'].dropna().sample(100).user_id.to_arrow().to_pylist()
start = time.time()
results = engine.recommend_batch(test_100, k=1000)
print(f"{time.time()-start:.2f}s total, {(time.time()-start)*10:.1f}ms/user")
