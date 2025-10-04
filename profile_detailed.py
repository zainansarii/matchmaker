"""
Detailed profiling to find remaining bottlenecks
"""
import time
import sys
sys.path.insert(0, '/home/zain/projects/matchmaker/src')

from matchmaker import matchmaker
import cudf
import numpy as np

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

user_df = engine.user_df
test_users = user_df[user_df.gender=='M'].dropna().sample(100).user_id.to_arrow().to_pylist()

print("\n" + "="*70)
print("DETAILED PROFILING - 100 users")
print("="*70)

# Profile metadata lookup
start = time.time()
user_ids_set = set(test_users)
if isinstance(engine.user_df, cudf.DataFrame):
    users_subset = engine.user_df[engine.user_df['user_id'].isin(user_ids_set)][['user_id', 'gender', 'league']].to_pandas()
else:
    users_subset = engine.user_df[engine.user_df['user_id'].isin(user_ids_set)][['user_id', 'gender', 'league']]
user_metadata = {row['user_id']: {'gender': row['gender'], 'league': row['league']} for _, row in users_subset.iterrows()}
metadata_time = (time.time() - start) * 1000
print(f"1. Metadata lookup: {metadata_time:.1f}ms")

# Profile just the recommender batch call
start = time.time()
batch_results = engine.recommender.recommend_batch(test_users, user_metadata, k=1000)
recommender_time = (time.time() - start) * 1000
print(f"2. Recommender.recommend_batch: {recommender_time:.1f}ms")

# Now profile internal components by instrumenting the code
print("\n--- Breaking down recommender.recommend_batch ---")

# Manually replicate what happens inside to profile each step
from collections import defaultdict

# Group users
start = time.time()
groups = {}
for uid in test_users:
    if uid not in user_metadata:
        continue
    meta = user_metadata[uid]
    gender = meta.get('gender')
    league = meta.get('league')
    key = (gender, league)
    if key not in groups:
        groups[key] = []
    groups[key].append(uid)
grouping_time = (time.time() - start) * 1000
print(f"  - Grouping by (gender, league): {grouping_time:.1f}ms")
print(f"    Groups: {[(k, len(v)) for k, v in groups.items()]}")

# Profile _batch_recommend_males for the largest group
largest_group = max(groups.items(), key=lambda x: len(x[1]))
(gender, league), male_ids = largest_group

if gender == 'M':
    print(f"\n  - Profiling largest group: {len(male_ids)} males in {league}")
    
    # Step 1: Filter valid males and get factors
    start = time.time()
    valid_pairs = []
    for mid in male_ids:
        if mid in engine.als_model.male_map:
            m_idx = engine.als_model.male_map[mid]
            male_factor = engine.als_model.male_factors[m_idx].get()  # GPU->CPU transfer
            valid_pairs.append((mid, male_factor))
    factor_extraction_time = (time.time() - start) * 1000
    print(f"    a) Extract factors (GPU->CPU): {factor_extraction_time:.1f}ms")
    
    # Step 2: Stack factors
    start = time.time()
    male_ids_ordered, factors_list = zip(*valid_pairs)
    query_matrix = np.vstack(factors_list).astype('float32')
    stacking_time = (time.time() - start) * 1000
    print(f"    b) Stack query matrix: {stacking_time:.1f}ms")
    
    # Step 3: FAISS search
    start = time.time()
    index = engine.recommender.female_indices[league]
    k_search = min(1000, len(engine.recommender.female_id_map[league]))
    distances, indices = index.search(query_matrix, k_search)
    faiss_time = (time.time() - start) * 1000
    print(f"    c) FAISS search ({len(male_ids_ordered)} x {k_search}): {faiss_time:.1f}ms")
    
    # Step 4: Collect pairs
    start = time.time()
    all_pairs = []
    for i, mid in enumerate(male_ids_ordered):
        for idx in indices[i]:
            if idx == -1:
                continue
            female_id = engine.recommender.female_id_map[league][idx]
            all_pairs.append((mid, female_id))
    pair_collection_time = (time.time() - start) * 1000
    print(f"    d) Collect pairs: {pair_collection_time:.1f}ms ({len(all_pairs)} pairs)")
    
    # Step 5: Batch mutual scoring
    start = time.time()
    all_scores = engine.als_model.mutual_score_batch(all_pairs)
    mutual_score_time = (time.time() - start) * 1000
    print(f"    e) Batch mutual scoring: {mutual_score_time:.1f}ms")
    
    # Step 6: Group and sort
    start = time.time()
    user_recommendations = {mid: [] for mid in male_ids_ordered}
    for (mid, fid), score in zip(all_pairs, all_scores):
        if score > 0:
            user_recommendations[mid].append((fid, float(score)))
    results = {}
    for mid in male_ids_ordered:
        recs = user_recommendations[mid]
        recs.sort(key=lambda x: x[1], reverse=True)
        results[mid] = recs[:1000]
    grouping_sorting_time = (time.time() - start) * 1000
    print(f"    f) Group and sort results: {grouping_sorting_time:.1f}ms")
    
    print(f"\n  Total for this group: {factor_extraction_time + stacking_time + faiss_time + pair_collection_time + mutual_score_time + grouping_sorting_time:.1f}ms")

print("\n" + "="*70)
print("BOTTLENECK ANALYSIS")
print("="*70)
print("⚠️  GPU->CPU transfers in factor extraction are expensive!")
print("    Solution: Keep factors on GPU, pass CuPy arrays to FAISS GPU")
