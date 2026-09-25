# Neuron Query Format Guide

This document describes the flexible neuron query format supported throughout the drocat project.

## Overview

The project supports two main query formats:
1. **Legacy format**: Lists of identifiers, regex patterns, or bodyIds
2. **Dict filter format**: Simple operator-based filters (same as `type_filter` in NeuronBridge)

Both formats work with:
- `FindNeuronConnection.sourceNeurons` and `targetNeurons`
- `statvis.get_types()`, `get_bodyIds()`, `get_instances()`, `get_info()`
- `statvis.getNeurons()`

Both formats accept a `search_columns` scope (see
[Restricting the Search Columns](#restricting-the-search-columns)) that
controls which columns are searched when resolving names.

## Legacy Format (List-based)

The traditional format uses lists of neuron identifiers:

```python
# Single type
sourceNeurons = ['aMe12']

# Multiple types
sourceNeurons = ['MBON01', 'MBON02', 'MBON03']

# Regex patterns
sourceNeurons = ['aMe.*']  # All types starting with 'aMe'
sourceNeurons = ['.*DN.*']  # All types containing 'DN'
sourceNeurons = ['Mi[1-9]']  # Mi1 through Mi9

# BodyIds
sourceNeurons = [12345, 67890]

# Mixed (types, patterns, bodyIds)
sourceNeurons = ['aMe12', 'Mi.*', 720575940610453042]

# All neurons
sourceNeurons = None

# All typed neurons
sourceNeurons = []
```

## Special Token: `all_neurons`

For the pathfinding entry points (`FindNeuronConnection` used by the
Complete Paths and Shortest Paths tools), the string token `'all_neurons'`
loads the full typed-neuron set on one side, so the run fetches all
adjacent neurons at the given thresholds:

```python
# All neurons that connect TO PPL101 at the given thresholds
fc = FindNeuronConnection(
    sourceNeurons=['all_neurons'],
    targetNeurons=['PPL101'],
)
```

Semantics (enforced in `InitializeNeuronInfo`, so the UI and direct
script/API callers behave identically):

- `'all_neurons'` is matched case-insensitively and **replaces every other
  chip** in the same query. Internally it resolves to an empty-list
  `getNeurons` query (all typed neurons from local data, works offline).
- Both sides = `'all_neurons'` is **not allowed** (raises `ValueError`).
- An `all_neurons` side **forces `max_interlayer = 0`** (direct connections
  only); a previous value is ignored with a warning.

## Restricting the Search Columns

When a name is resolved (e.g. `'MTe07'`), the legacy format searches columns
in a fixed priority: `bodyId` → `type` → `instance` → every other string
column (such as `flywireType`, `hemibrainType`, `mancType`). The final
"other columns" step runs for **both exact and regex** names — before the
2026-08 fix it ran for regex patterns only, so an exact name that lived in
a cross-dataset column (e.g. `MTe07` in `flywireType` on male-cns v1.0)
was silently reported as "not found in any column".

Use the `search_columns` parameter to control the scope:

| Value | Searches | Use case |
| --- | --- | --- |
| `'auto'` (default) | All columns, priority `bodyId` → `type` → `instance` → other string columns | Names that may live in cross-dataset type columns (`MTe07` in `flywireType`) |
| `'type'` | Only the `type` column | Type-only queries; avoids false hits in other columns |
| `'instance'` | Only the `instance` column | Instance-only queries |
| `'bodyId'` | Only the `bodyId` column | BodyId-only queries |

```python
# MTe07 exists only in flywireType on male-cns v1.0 -> found with 'auto'
fc = FindNeuronConnection(
    dataset='male-cns:v1.0',
    sourceNeurons=['MTe07'],
    search_columns='auto',   # default
)

# Restrict to the native CNS type column
fc = FindNeuronConnection(
    dataset='male-cns:v1.0',
    sourceNeurons=['MBON01'],
    search_columns='type',
)
```

The UI exposes the same options in the **Search Columns** selector of the
Complete Paths, Shortest Paths, Network and Cross-Dataset Comparison tabs
(`auto` / `type` / `instance` / `bodyId`).

## Dict Filter Format (Same as type_filter)

The dict filter format is the same simple format used by `type_filter` in NeuronBridge scripts.
It **auto-searches** across type, instance, bodyId, and other string columns.

### Basic Syntax

```python
{'operator': value_or_list}
```

### Supported Operators

| Operator       | Description           | Example                     |
| -------------- | --------------------- | --------------------------- |
| `contains`     | Substring match       | `{'contains': 'DN'}`        |
| `startswith`   | Prefix match          | `{'startswith': 'aMe'}`     |
| `endswith`     | Suffix match          | `{'endswith': '_R'}`        |
| `regex`        | Full regex pattern    | `{'regex': r'DN[a-z]\d+'}`  |
| `exact`        | Exact value match     | `{'exact': ['Mi1', 'Mi2']}` |
| `not_contains` | Exclude substring     | `{'not_contains': 'test'}`  |
| `not_regex`    | Exclude regex pattern | `{'not_regex': r'.*_L$'}`   |

### Multiple Values (OR logic)

When a single operator has multiple values (list), they use OR logic:

```python
# Matches neurons with type/instance starting with 'aMe' OR 'Mi' OR 'Tm'
{'startswith': ['aMe', 'Mi', 'Tm']}

# Matches neurons containing 'DN' OR 'AN'
{'contains': ['DN', 'AN']}
```

### Multiple Operators (AND logic)

When combining different operators, they use AND logic:

```python
# Neurons starting with 'DN' AND containing 'a'
{'startswith': 'DN', 'contains': 'a'}

# Neurons containing 'DN' AND ending with '_R'
{'contains': 'DN', 'endswith': '_R'}
```

### Auto-Search Column Priority

The filter automatically searches columns in this priority:
1. `type` - Neuron type name
2. `instance` - Instance name (often includes _L/_R)
3. `bodyId` - Unique neuron identifier (converted to string)
4. Other string columns in the DataFrame

A match in **any** column satisfies the filter.

## Examples

### FindNeuronConnection

```python
from src.coana import FindNeuronConnection

# Legacy format
fc = FindNeuronConnection(
    sourceNeurons=['aMe.*'],
    targetNeurons=['.*DN.*']
)

# Dict filter format (simpler, same as type_filter)
fc = FindNeuronConnection(
    sourceNeurons={'contains': 'aMe'},
    targetNeurons={'contains': 'DN'}
)

# Combined filter (AND logic)
fc = FindNeuronConnection(
    sourceNeurons={'startswith': ['DN', 'AN']},  # OR within list
    targetNeurons={'contains': 'MBON', 'endswith': '_R'}  # AND across operators
)
```

### statvis Query Functions

```python
from src import statvis as sv

# Simple return (recommended)
types = sv.get_types({'contains': 'DN'}, return_simple=True)

# Get bodyIds matching criteria
bodyIds = sv.get_bodyIds(
    {'startswith': 'aMe'},
    dataset='hemibrain:v1.2.1',
    return_simple=True
)

# Get full neuron info
df = sv.get_info({'contains': 'DN'}, dataset='male-cns:v0.9')

# Legacy format still works
types, map_dict, ds = sv.get_types('aMe.*', dataset='hemibrain:v1.2.1')
```

### Simplified Return Values

The query functions now support `return_simple=True` for cleaner code:

```python
# Old way (still works)
type_list, map_dict, ds = sv.get_types('aMe.*')

# New way (cleaner)
type_list = sv.get_types('aMe.*', return_simple=True)

# Works with dict filters too
type_list = sv.get_types({'contains': 'DN'}, return_simple=True)
```

## Migration Guide

### From Regex to Dict Filter

| Legacy          | Dict Filter                                     |
| --------------- | ----------------------------------------------- |
| `['aMe.*']`     | `{'startswith': 'aMe'}` or `{'regex': 'aMe.*'}` |
| `['.*DN.*']`    | `{'contains': 'DN'}`                            |
| `['.*_R']`      | `{'endswith': '_R'}`                            |
| `['DN[ab]\d+']` | `{'regex': r'DN[ab]\d+'}`                       |

### Benefits of Dict Filter

1. **Readability**: Clear intent without regex knowledge
2. **Combine filters**: Easy AND/OR logic
3. **Less error-prone**: No regex escaping issues
4. **Self-documenting**: Filter structure shows what's matched
5. **Auto-search**: Automatically finds matching columns

## Technical Details

The `NeuronFilter` class in `src/utils/neuron_filter.py` handles both formats:

```python
from src.utils.neuron_filter import NeuronFilter

# Create filter from either format
nf = NeuronFilter(['aMe.*'])  # Legacy
nf = NeuronFilter({'contains': 'DN'})  # Dict filter (auto-searches columns)

# Apply to DataFrame
matched_df = nf.apply(neuron_df)
bodyIds = nf.get_bodyIds(neuron_df)
types = nf.get_types(neuron_df)

# Get description
print(nf.describe())  # "contains['DN']"
```
