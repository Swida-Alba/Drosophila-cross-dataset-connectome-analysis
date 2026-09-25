# Co-Labeling Similarity Methods

## Overview

The NeuronBridge FindLines analysis now supports three different similarity metrics for comparing driver lines based on the cell types they label:

1. **Binary Jaccard** (original): Treats all labeled types equally (present/absent)
2. **Weighted Jaccard** (new): Incorporates match scores into similarity calculation
3. **Rank Correlation** (new): Measures correlation of type rankings based on match scores

## Implementation Details

### 1. Binary Jaccard Similarity

**Formula:**
```
J(A, B) = |Types_A ∩ Types_B| / |Types_A ∪ Types_B|
```

**Characteristics:**
- Original implementation
- Treats all cell types equally (binary: present or absent)
- Range: [0, 1] where 1 = identical type sets

**Use when:** You want a simple measure of type overlap without considering match quality

### 2. Weighted Jaccard Similarity

**Formula:**
```
WJ(A, B) = Σ min(score_A(t), score_B(t)) / Σ max(score_A(t), score_B(t))
```
where the sums are over all types t in A ∪ B

**Characteristics:**
- Uses match scores as weights
- Higher-scoring matches contribute more to similarity
- Lines with strong matches to the same types score higher
- Range: [0, 1] where 1 = identical type-score distributions

**Use when:** You want to emphasize lines with strong matches to common cell types

**Example:**
```
Line A: {type1: 0.9, type2: 0.3}
Line B: {type1: 0.8, type2: 0.7}

Intersection: min(0.9, 0.8) + min(0.3, 0.7) = 0.8 + 0.3 = 1.1
Union:        max(0.9, 0.8) + max(0.3, 0.7) = 0.9 + 0.7 = 1.6
Weighted Jaccard = 1.1 / 1.6 = 0.6875
```

### 3. Rank Correlation Similarity

**Formula:**
```
ρ(A, B) = Spearman correlation of scores for types in A ∩ B
```

**Characteristics:**
- Only considers types labeled by BOTH lines
- Measures whether lines rank shared types similarly
- Insensitive to score scaling (uses ranks only)
- Range: [-1, 1] where 1 = perfect agreement in rankings, -1 = perfect disagreement

**Use when:** You want to find lines that prioritize the same cell types (regardless of absolute scores)

**Example:**
```
Line A: {type1: 0.9, type2: 0.5, type3: 0.3}
Line B: {type1: 0.8, type2: 0.6, type3: 0.2}

Common types: {type1, type2, type3}
Ranks in A:   {1, 2, 3}
Ranks in B:   {1, 2, 3}
Spearman correlation = 1.0 (perfect agreement)
```

## Usage

### In FindLines Batch

The `find_lines_batch()` method now automatically generates all three similarity matrices when calculating specificity:

```python
from neuronbridge_finder import NeuronBridgeFinder

nbf = NeuronBridgeFinder(verbose=True)

results = nbf.find_lines_batch(
    queries='MBON01',
    dataset='hemibrain:v1.2.1',
    output_dir='./output',
    calculate_specificity=True,
    specificity_top_n=50  # Top 50 lines will have co-labeling analysis
)
```

**Output files:**
- `colabeling_matrix_jaccard.csv` - Binary Jaccard matrix
- `colabeling_matrix_jaccard.html` - Interactive heatmap
- `colabeling_matrix_weighted_jaccard.csv` - Weighted Jaccard matrix
- `colabeling_matrix_weighted_jaccard.html` - Interactive heatmap
- `colabeling_matrix_rank_correlation.csv` - Rank correlation matrix
- `colabeling_matrix_rank_correlation.html` - Interactive heatmap

### Direct Method Call

You can also call `_build_colabeling_matrix()` directly with a specific method:

```python
from neuronbridge_finder import NeuronBridgeFinder

nbf = NeuronBridgeFinder(verbose=True)

lines = ['LH173', 'LH174', 'LH175']

# Weighted Jaccard
matrix, type_sets = nbf._build_colabeling_matrix(
    lines=lines,
    match_type='cds',
    top_n=100,
    similarity_method='weighted_jaccard'
)

# Rank correlation
matrix, type_sets = nbf._build_colabeling_matrix(
    lines=lines,
    match_type='cds',
    top_n=100,
    similarity_method='rank_correlation'
)
```

## Interpretation Guide

### Comparing Methods

**High Binary Jaccard + High Weighted Jaccard:**
- Lines label many common types
- Matches to common types are strong
- **Interpretation:** Lines are highly similar

**High Binary Jaccard + Low Weighted Jaccard:**
- Lines label many common types
- BUT matches to common types are weak
- **Interpretation:** Lines overlap but with poor-quality matches

**Low Binary Jaccard + High Rank Correlation:**
- Lines label few common types
- BUT they rank those common types similarly
- **Interpretation:** Lines have different breadth but agree on priorities

**High Weighted Jaccard + Low Rank Correlation:**
- Lines have strong matches to common types
- BUT rank them differently
- **Interpretation:** Lines label similar types with different priorities

### Choosing a Method

| Goal | Recommended Method |
|------|-------------------|
| Find lines with similar type coverage | Binary Jaccard |
| Find lines with strong matches to same types | Weighted Jaccard |
| Find lines that prioritize same types | Rank Correlation |
| Comprehensive comparison | Use all three methods |

## Technical Notes

### Sparsity Calculations

The sparsity metrics (used in `line_stats.csv`) are now based on **weighted Jaccard** as it provides the most informative measure of functional similarity.

### Score Handling

- Scores are extracted from the `line_to_neuron()` results
- For types appearing multiple times, the **maximum score** is retained
- Missing types (in union for weighted Jaccard) have score = 0.0

### Correlation Edge Cases

- If lines share < 2 types, rank correlation = 0.0
- If scores are constant (all equal), rank correlation = 0.0 (NaN is converted)

## Performance

All three matrices are computed in a single pass through the lines:
- Time complexity: O(n² × t) where n = number of lines, t = types per line
- Memory: O(n × t) for type-score dictionaries

For 50 lines with ~100 types each, computation takes ~30-60 seconds (dominated by API calls).

## Examples

See `tests/core/test_neuronbridge_colabeling.py` for a complete example testing all three methods.
