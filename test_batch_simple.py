"""
Lightweight test - just check mutual_score_batch performance
"""
import time
import sys
sys.path.insert(0, '/home/zain/projects/matchmaker/src')

import numpy as np
import cupy as cp
from matchmaker.models.als import ALSModel

print("Testing mutual_score_batch performance...")

# Create a small test ALS model
model = ALSModel(factors=64)

# Mock some factors for testing
n_males = 100
n_females = 100
model.male_factors = cp.random.randn(n_males, 64).astype(np.float32)
model.female_factors = cp.random.randn(n_females, 64).astype(np.float32)
model.female_pref_factors = cp.random.randn(n_females, 64).astype(np.float32)
model.male_attr_factors = cp.random.randn(n_males, 64).astype(np.float32)

# Create mappings
model.male_map = {i: i for i in range(n_males)}
model.female_map = {i: i for i in range(n_females)}
model.male_map_f2m = {i: i for i in range(n_males)}
model.female_map_f2m = {i: i for i in range(n_females)}

# Test pairs
test_pairs = [(i % n_males, i % n_females) for i in range(1000)]

# Time batch scoring
start = time.time()
scores = model.mutual_score_batch(test_pairs)
batch_time = (time.time() - start) * 1000
print(f"✅ Batch scored 1000 pairs in {batch_time:.1f}ms")

# Time individual scoring for comparison
start = time.time()
for m, f in test_pairs[:100]:
    _ = model.mutual_score(m, f)
single_time = (time.time() - start) * 10  # x10 to extrapolate
print(f"⚠️  Individual (100 pairs x10): ~{single_time:.1f}ms")

print(f"\nSpeedup: {single_time/batch_time:.1f}x faster with batching")
