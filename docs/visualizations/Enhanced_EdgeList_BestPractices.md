# Enhanced Edge-List Format - Best Practices & Tips

## Quick Start

### Simplest Possible Usage

```python
import pandas as pd
from vispath import VisualizePath

# Create edge-list (minimum 3 columns)
df = pd.DataFrame({
    'source': ['A', 'B', 'C'],
    'target': ['B', 'C', 'D'],
    'weight': [10, 15, 20]
})

# Visualize
vis = VisualizePath(path_file=df, output_folder='./output')
vis.visualize()  # Creates heatmap, sankey, and network
```

---

## Best Practices

### 1. Column Naming

#### ✅ Recommended

Use consistent, meaningful names:

```python
# For neuron body IDs
{'bodyId_pre': [...], 'bodyId_post': [...], 'synapse_count': [...]}

# For neuron types
{'type_pre': [...], 'type_post': [...], 'weight': [...]}

# For general neurons
{'neuron_pre': [...], 'neuron_post': [...], 'weight': [...]}
```

#### ⚠️ Avoid

Mixing naming conventions in the same dataset:

```python
# DON'T: Inconsistent naming
{'bodyId_pre': [...], 'target': [...], 'weight': [...]}  # Confusing!

# DO: Consistent naming
{'bodyId_pre': [...], 'bodyId_post': [...], 'weight': [...]}  # Clear!
```

### 2. Metric Organization

#### ✅ Recommended

Place standard metrics first, custom metrics after:

```python
data = {
    'source': [...],
    'target': [...],
    'weight': [...],           # Required
    'ratio': [...],            # Standard metric (toggleable)
    'probability': [...],      # Standard metric (toggleable)
    'strength': [...],         # Custom metric
    'confidence': [...]        # Custom metric
}
```

#### 💡 Why?

- Easier to read and understand
- Standard metrics get toggle controls
- Custom metrics preserved for analysis

### 3. File Organization

#### ✅ Recommended

```
project/
├── data/
│   ├── edges_raw.csv          # Raw edge data
│   ├── edges_processed.csv    # With metrics
│   └── metadata.json          # Column descriptions
├── output/
│   ├── network/              # Network visualizations
│   ├── sankey/               # Sankey diagrams
│   └── heatmap/              # Heatmaps
└── scripts/
    └── visualize.py          # Visualization script
```

### 4. Data Validation

#### ✅ Always Validate

```python
import pandas as pd

# Load data
df = pd.read_csv('edges.csv')

# Check for required columns
required_cols = ['source', 'target', 'weight']  # Or your naming convention
missing = [col for col in required_cols if col not in df.columns]
if missing:
    print(f"❌ Missing columns: {missing}")
else:
    print(f"✅ All required columns present")

# Check for numeric weight
if not pd.api.types.is_numeric_dtype(df['weight']):
    print("⚠️ Warning: Weight column is not numeric")

# Check for null values
nulls = df[required_cols].isnull().sum()
if nulls.any():
    print(f"⚠️ Warning: Null values found:\n{nulls}")

# Check for duplicates
dupes = df.duplicated(subset=['source', 'target']).sum()
if dupes > 0:
    print(f"⚠️ Warning: {dupes} duplicate edges found")
```

---

## Common Patterns

### Pattern 1: Basic Network

```python
from vispath import VisualizePath

# Simple network with just edges
df = pd.DataFrame({
    'source': ['A', 'B', 'C'],
    'target': ['B', 'C', 'D'],
    'weight': [10, 15, 20]
})

vis = VisualizePath(path_file=df, output_folder='./output')
vis.create_network()
```

### Pattern 2: Network with Connection Ratios

```python
# Add ratio for proportion-based filtering
df = pd.DataFrame({
    'source': ['KC1', 'KC1', 'KC2'],
    'target': ['MBON1', 'MBON2', 'MBON1'],
    'weight': [100, 50, 80],
    'ratio': [0.67, 0.33, 1.0]  # Auto-toggleable
})

vis = VisualizePath(path_file=df, output_folder='./output')
vis.create_network()  # Toggle ratio on/off in browser
```

### Pattern 3: Full Pipeline with All Metrics

```python
# Complete dataset with multiple metrics
df = pd.DataFrame({
    'bodyId_pre': [123456, 234567, 345678],
    'bodyId_post': [234567, 345678, 456789],
    'synapse_count': [150, 120, 100],
    'ratio': [0.6, 0.5, 0.4],
    'probability': [0.95, 0.88, 0.92],
    'strength': [0.85, 0.82, 0.91],
    'confidence': [0.92, 0.88, 0.95]
})

vis = VisualizePath(path_file=df, output_folder='./output')
vis.visualize()  # Creates all three visualizations with toggles
```

### Pattern 4: Batch Processing

```python
import os
from vispath import VisualizePath

# Process multiple datasets
datasets = {
    'KC_to_MBON': 'data/kc_mbon_edges.csv',
    'MBON_to_DAN': 'data/mbon_dan_edges.csv',
    'DAN_to_MBON': 'data/dan_mbon_edges.csv'
}

for name, filepath in datasets.items():
    print(f"\nProcessing: {name}")
    vis = VisualizePath(
        path_file=filepath,
        output_folder=f'./output/{name}'
    )
    vis.visualize()
    print(f"✓ {name} complete")
```

### Pattern 5: From NeuPrint Query

```python
from neuprint import Client, fetch_adjacencies
from vispath import VisualizePath

# Query NeuPrint
c = Client('neuprint.janelia.org', dataset='hemibrain:v1.2.1')
edges = fetch_adjacencies(None, None)  # Or specific criteria

# Convert to edge-list format
df = edges[['bodyId_pre', 'bodyId_post', 'weight']].copy()

# Add ratio (optional)
df['ratio'] = df.groupby('bodyId_pre')['weight'].transform(
    lambda x: x / x.sum()
)

# Visualize
vis = VisualizePath(path_file=df, output_folder='./output')
vis.visualize()
```

---

## Tips & Tricks

### Tip 1: Column Name Detection

Use suffix matching for flexibility:

```python
# All of these work with *_pre/*_post pattern:
'bodyId_pre', 'bodyId_post'      # ✓
'type_pre', 'type_post'          # ✓
'neuron_pre', 'neuron_post'      # ✓
'cell_pre', 'cell_post'          # ✓
'node_pre', 'node_post'          # ✓
```

### Tip 2: Weight Column Aliases

Multiple weight column names supported:

```python
'weight'          # ✓ Standard
'weights'         # ✓ Plural
'synapse_count'   # ✓ Neuroscience
'count'           # ✓ Simple
```

### Tip 3: Ratio Calculation

Calculate connection ratios from weights:

```python
import pandas as pd

df = pd.DataFrame({
    'source': ['A', 'A', 'B'],
    'target': ['X', 'Y', 'X'],
    'weight': [100, 50, 80]
})

# Add ratio (proportion of output)
df['ratio'] = df.groupby('source')['weight'].transform(
    lambda x: x / x.sum()
)

print(df)
#   source target  weight  ratio
# 0      A      X     100   0.67
# 1      A      Y      50   0.33
# 2      B      X      80   1.00
```

### Tip 4: Probability from Ratio

Convert ratio to probability:

```python
# Simple threshold
df['probability'] = df['ratio'].apply(lambda x: 0.9 if x > 0.5 else 0.7)

# Or more sophisticated
import numpy as np
df['probability'] = 1 - np.exp(-df['ratio'])  # Exponential
```

### Tip 5: Custom Metric Names

Use descriptive names for custom metrics:

```python
# ✅ Good: Descriptive
'signal_strength'
'connection_reliability'
'transmission_efficiency'

# ⚠️ Avoid: Ambiguous
'value1'
'metric'
'data'
```

### Tip 6: Excel Multi-Sheet Handling

Process multiple sheets:

```python
import pandas as pd

# Read all sheets
excel_file = pd.ExcelFile('data.xlsx')
for sheet_name in excel_file.sheet_names:
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    vis = VisualizePath(
        path_file=df,
        output_folder=f'./output/{sheet_name}'
    )
    vis.visualize()
```

### Tip 7: Data Export

Export processed data with metrics:

```python
vis = VisualizePath(path_file=df, output_folder='./output')
vis.visualize()

# Data is automatically saved as:
# ./output/[name]_data.xlsx

# All columns preserved:
# - source, target, weight
# - Standard metrics (ratio, probability)
# - Custom metrics
# - Path information
```

---

## Troubleshooting

### Issue 1: Column Not Found

**Error**: "Could not find source/target column"

**Solution**:
```python
# Check actual column names
print(df.columns.tolist())

# Rename if needed
df = df.rename(columns={
    'from_neuron': 'source',
    'to_neuron': 'target',
    'count': 'weight'
})
```

### Issue 2: Weight Not Numeric

**Error**: Weight calculations fail

**Solution**:
```python
# Convert to numeric
df['weight'] = pd.to_numeric(df['weight'], errors='coerce')

# Remove non-numeric rows
df = df.dropna(subset=['weight'])
```

### Issue 3: Duplicate Edges

**Problem**: Multiple rows for same edge

**Solution**:
```python
# Option 1: Sum weights
df = df.groupby(['source', 'target']).agg({
    'weight': 'sum',
    'ratio': 'mean',      # Or 'max', 'min'
    'probability': 'mean'
}).reset_index()

# Option 2: Keep first
df = df.drop_duplicates(subset=['source', 'target'], keep='first')
```

### Issue 4: Missing Metrics

**Problem**: Ratio/probability not detected

**Solution**:
```python
# Check column names (case-insensitive)
print(df.columns.tolist())

# Rename if needed
df = df.rename(columns={
    'Ratio': 'ratio',              # Lowercase
    'Probability': 'probability'   # Lowercase
})
```

### Issue 5: Large Datasets

**Problem**: Slow processing for >10,000 edges

**Solution**:
```python
# Filter before visualization
threshold = 10  # Minimum weight
df_filtered = df[df['weight'] >= threshold]

# Or sample
df_sample = df.sample(n=5000, weights='weight', random_state=42)

# Then visualize
vis = VisualizePath(path_file=df_filtered, output_folder='./output')
```

---

## Performance Tips

### 1. Pre-filter Data

```python
# Filter before creating VisualizePath
df_large = pd.read_csv('large_dataset.csv')
df_filtered = df_large[df_large['weight'] >= 5]  # Only strong connections

vis = VisualizePath(path_file=df_filtered, output_folder='./output')
```

### 2. Use Appropriate Data Types

```python
# Optimize data types
df['source'] = df['source'].astype('category')
df['target'] = df['target'].astype('category')
df['weight'] = df['weight'].astype('int32')  # Or float32 if needed
```

### 3. Batch Process in Parallel

```python
from concurrent.futures import ProcessPoolExecutor

def process_dataset(args):
    name, filepath = args
    vis = VisualizePath(path_file=filepath, output_folder=f'./output/{name}')
    vis.visualize()
    return name

datasets = [
    ('dataset1', 'data1.csv'),
    ('dataset2', 'data2.csv'),
    ('dataset3', 'data3.csv')
]

with ProcessPoolExecutor(max_workers=3) as executor:
    results = executor.map(process_dataset, datasets)
    for result in results:
        print(f"✓ {result} complete")
```

---

## Migration Guide

### From Rigid Format

**Before**:
```python
# Required exact column names
df = pd.DataFrame({
    'source': ['A', 'B'],
    'target': ['B', 'C'],
    'weight': [10, 15]
})
```

**After**:
```python
# Flexible naming - any of these work:
df = pd.DataFrame({'pre': [...], 'post': [...], 'count': [...]})
df = pd.DataFrame({'from': [...], 'to': [...], 'weights': [...]})
df = pd.DataFrame({'bodyId_pre': [...], 'bodyId_post': [...], 'synapse_count': [...]})
```

### Adding Metrics

**Before**:
```python
# Manual metric creation
vis = VisualizePath(path_file=df, output_folder='./output')
```

**After**:
```python
# Just add numeric columns - auto-detected
df['ratio'] = [...]           # Auto-toggleable
df['probability'] = [...]     # Auto-toggleable
df['custom_metric'] = [...]   # Preserved for export
```

---

## See Also

- [Enhanced_EdgeList_Format.md](Enhanced_EdgeList_Format.md) - Complete documentation
- [Enhanced_EdgeList_QuickRef.md](Enhanced_EdgeList_QuickRef.md) - Quick reference
- [examples/Example_SimpleEdgeList.py](../../archive/examples/visualization/input_formats/Example_SimpleEdgeList.py) - Working examples
- [tests/ui/test_edge_list_editor.py](../../tests/ui/test_edge_list_editor.py) - Test suite (the edge-list editor)
