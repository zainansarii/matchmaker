"""
Profile just the mutual_score_batch to find what's slow
"""
import time
import sys
sys.path.insert(0, '/home/zain/projects/matchmaker/src')

import numpy as np
import cupy as cp
from matchmaker.models.als import ALSModel

# Create test model
model = ALSModel(factors=64)
n_males = 1000
n_females = 1000
model.male_factors = cp.random.randn(n_males, 64).astype(np.float32)
model.female_factors = cp.random.randn(n_females, 64).astype(np.float32)
model.female_pref_factors = cp.random.randn(n_females, 64).astype(np.float32)
model.male_attr_factors = cp.random.randn(n_males, 64).astype(np.float32)
model.male_map = {i: i for i in range(n_males)}
model.female_map = {i: i for i in range(n_females)}
model.male_map_f2m = {i: i for i in range(n_males)}
model.female_map_f2m = {i: i for i in range(n_females)}

# Test with realistic batch size (44 users × 1000 candidates = 44000 pairs)
test_pairs = [(i % n_males, (i * 7) % n_females) for i in range(44000)]

print("Profiling mutual_score_batch internals...")
print(f"Test size: {len(test_pairs)} pairs\n")

# Manual breakdown of mutual_score_batch
start = time.time()
pairs_array = np.array(test_pairs, dtype=np.int32)
male_ids = pairs_array[:, 0]
female_ids = pairs_array[:, 1]
array_conversion_time = (time.time() - start) * 1000
print(f"1. Convert to numpy arrays: {array_conversion_time:.1f}ms")

start = time.time()
scores_gpu = cp.zeros(len(test_pairs), dtype=cp.float32)
alloc_time = (time.time() - start) * 1000
print(f"2. Allocate GPU array: {alloc_time:.1f}ms")

# M2F lookups
start = time.time()
m2f_male_valid = np.array([m in model.male_map for m in male_ids])
m2f_female_valid = np.array([f in model.female_map for f in female_ids])
m2f_valid = m2f_male_valid & m2f_female_valid
lookup1_time = (time.time() - start) * 1000
print(f"3. M2F validity checks (Python dicts): {lookup1_time:.1f}ms  ⚠️ SLOW!")

start = time.time()
valid_idx = np.where(m2f_valid)[0]
m_factor_idx = np.array([model.male_map[male_ids[i]] for i in valid_idx])
f_factor_idx = np.array([model.female_map[female_ids[i]] for i in valid_idx])
index_lookup_time = (time.time() - start) * 1000
print(f"4. M2F index lookups: {index_lookup_time:.1f}ms  ⚠️ SLOW!")

start = time.time()
m_factors = model.male_factors[m_factor_idx]
f_factors = model.female_factors[f_factor_idx]
m2f_scores = cp.sum(m_factors * f_factors, axis=1)
scores_gpu[valid_idx] += m2f_scores
gpu_compute_time = (time.time() - start) * 1000
print(f"5. GPU compute (gather + multiply + sum): {gpu_compute_time:.1f}ms ✅")

# F2M (similar)
start = time.time()
f2m_female_valid = np.array([f in model.female_map_f2m for f in female_ids])
f2m_male_valid = np.array([m in model.male_map_f2m for m in male_ids])
f2m_valid = f2m_female_valid & f2m_male_valid
valid_idx = np.where(f2m_valid)[0]
f_pref_idx = np.array([model.female_map_f2m[female_ids[i]] for i in valid_idx])
m_attr_idx = np.array([model.male_map_f2m[male_ids[i]] for i in valid_idx])
lookup2_time = (time.time() - start) * 1000
print(f"6. F2M lookups: {lookup2_time:.1f}ms  ⚠️ SLOW!")

start = time.time()
f_factors = model.female_pref_factors[f_pref_idx]
m_factors = model.male_attr_factors[m_attr_idx]
f2m_scores = cp.sum(f_factors * m_factors, axis=1)
scores_gpu[valid_idx] += f2m_scores
gpu_compute2_time = (time.time() - start) * 1000
print(f"7. GPU compute (F2M): {gpu_compute2_time:.1f}ms ✅")

start = time.time()
result = scores_gpu.get()
transfer_time = (time.time() - start) * 1000
print(f"8. GPU->CPU transfer: {transfer_time:.1f}ms")

total = array_conversion_time + alloc_time + lookup1_time + index_lookup_time + gpu_compute_time + lookup2_time + gpu_compute2_time + transfer_time
print(f"\nTotal: {total:.1f}ms")
print(f"\n⚠️ BOTTLENECK: Dictionary lookups in Python! ({lookup1_time + index_lookup_time + lookup2_time:.1f}ms)")
print("   The dict lookups are iterating 44K times in Python - need vectorization!")
