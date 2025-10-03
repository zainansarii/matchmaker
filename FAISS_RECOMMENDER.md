# League-Filtered FAISS Recommender

## Overview

The `LeagueFilteredRecommender` provides scalable, GPU-accelerated recommendations for dating apps by:

1. **League filtering**: Only matches users within the same attractiveness tier
2. **FAISS ANN search**: Fast approximate nearest neighbor search on ALS factor embeddings
3. **Mutual scoring**: Combines M2F and F2M scores for bidirectional compatibility

## Architecture

```
User → Get League → FAISS Search (same league only) → Top-K candidates → Mutual Score → Ranked Recommendations
```

### Key Components

- **FAISS Indices**: One index per (gender, league) combination
  - Males: 5 indices (Bronze/Silver/Gold/Platinum/Diamond males)
  - Females: 5 indices (Bronze/Silver/Gold/Platinum/Diamond females)
  
- **Index Types**:
  - `IndexFlatIP`: Exact inner product search for <100K users per league
  - `IndexIVFFlat`: Approximate search with IVF clustering for >100K users

- **GPU Acceleration**: Indices run on GPU when available (faiss-gpu)

## Usage

### Basic Usage

```python
from matchmaker import matchmaker

# Initialize and load data
engine = matchmaker.MatchingEngine()
engine.load_interactions("data/swipes.csv", ...)

# Run pipeline (builds FAISS indices)
engine.run_engagement()
engine.run_popularity()  # This builds the recommender!

# Get recommendations for a single user
recs = engine.recommend_for_user(user_id=12345, k=1000)
# Returns: [(candidate_id, mutual_score), ...] sorted by score descending

# Batch recommendations
batch_recs = engine.recommend_batch(user_ids=[123, 456, 789], k=1000)
# Returns: {user_id: [(candidate_id, mutual_score), ...], ...}
```

### Advanced Usage

```python
from matchmaker.models.recommender import LeagueFilteredRecommender

# Direct access to recommender
recommender = LeagueFilteredRecommender(
    als_model=engine.als_model,
    user_df=engine.user_df,
    use_gpu=True
)

# Gender-specific recommendations
male_recs = recommender.recommend_for_male(
    male_id=12345, 
    league='Gold', 
    k=1000
)

female_recs = recommender.recommend_for_female(
    female_id=67890,
    league='Platinum',
    k=1000
)
```

## Performance

### Scalability
- **Single query**: ~1-5ms per user (GPU)
- **Batch queries**: ~100-500 users/second (GPU)
- **Memory**: ~4 bytes per dimension per user (50 dims = 200 bytes/user)

### Example Benchmark (33K males, 33K females)
```
League      | Users | Index Type  | Query Time
------------|-------|-------------|------------
Bronze      | 10K   | IndexFlatIP | 1.2ms
Silver      | 10K   | IndexFlatIP | 1.3ms
Gold        | 8K    | IndexFlatIP | 1.0ms
Platinum    | 3K    | IndexFlatIP | 0.8ms
Diamond     | 2K    | IndexFlatIP | 0.6ms
```

## Scaling to Millions

For millions of users:

1. **Use IVF indices** (automatically enabled for >100K users/league)
2. **Tune nprobe**: Balance speed vs accuracy
   ```python
   # In _create_index():
   index.nprobe = 32  # Lower = faster, higher = more accurate
   ```

3. **Shard by region/market**: Build separate indices per geographic region

4. **Quantization**: Use PQ (Product Quantization) for memory efficiency
   ```python
   # For 10M+ users:
   index = faiss.IndexIVFPQ(quantizer, d, nlist, m=8, nbits=8)
   ```

## Installation

Add to `environment.yml`:
```yaml
channels:
  - pytorch
dependencies:
  - faiss-gpu=1.7.4
  - mkl=2021
  - blas=1.0=mkl
```

Or install manually:
```bash
conda install -c pytorch faiss-gpu=1.7.4
```

## Implementation Details

### League Filtering
- Each FAISS index contains only users from one league
- Prevents cross-league matches while maintaining speed
- Leagues assigned via PageRank-based balanced quantiles

### Mutual Score Calculation
```python
mutual_score = M2F_score + F2M_score
```
- M2F: How much male likes female (from male→female ALS)
- F2M: How much female likes male (from female→male ALS)
- Handles partial coverage (e.g., only M2F available)

### Index Building
1. Filter users by gender and league
2. Extract factor embeddings from ALS model
3. Build FAISS index (exact or approximate)
4. Move to GPU if available
5. Store ID mappings for result translation

## Future Enhancements

1. **Diversity**: Add MMR (Maximal Marginal Relevance) for diverse recommendations
2. **Filters**: Add additional constraints (age, location, preferences)
3. **Real-time updates**: Incremental index updates for new users
4. **Multi-league**: Allow ±1 league flexibility with score penalties
5. **Explainability**: Return factor contribution to scores

## References

- [FAISS Documentation](https://github.com/facebookresearch/faiss/wiki)
- [FAISS GPU Guide](https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU)
- [Implicit ALS](https://implicit.readthedocs.io/)
