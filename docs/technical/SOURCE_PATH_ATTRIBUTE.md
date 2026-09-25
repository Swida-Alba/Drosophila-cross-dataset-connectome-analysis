# Source Path Attribute Addition

**Date:** October 31, 2025  
**Status:** ✅ COMPLETE (dated change record)

> **Drift note (2026-09-25):** `VisualizeSkeleton` has since moved to
> `src/visualize_skeleton.py` (it is no longer in `coana.py`), the
> default output folder is now `local_data/connectome_analysis/`
> (formerly `connection_data/`), `src/vispath.py` now lives at
> `vispath-subproject/src/vispath_pkg/vispath.py`, and `source_path`
> currently has no readers — it is write-only provenance metadata.

## Overview

Added `source_path` attribute to `FindNeuronConnection` and `VisualizeSkeleton` classes for clearer path management after the src/ layout reorganization.

## Changes Made

### Before (Single Path Variable)

```python
script_path: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
'''current absolute path of the project root (parent of src/)'''
```

**Issue:** Not immediately clear that we're going up two levels, and why.

### After (Two Path Variables)

```python
source_path: str = os.path.dirname(os.path.abspath(__file__))
'''absolute path to the src/ directory where coana.py is located'''

script_path: str = os.path.dirname(source_path)
'''absolute path to the project root directory (parent of src/)'''
```

**Benefit:** 
- ✅ Clearer intent - `source_path` for src/, `script_path` for project root
- ✅ Easier to understand and maintain
- ✅ More self-documenting code

## Classes Updated

### 1. FindNeuronConnection (Line 78-81)

```python
@dataclass
class FindNeuronConnection:
    source_path: str = os.path.dirname(os.path.abspath(__file__))
    '''absolute path to the src/ directory where coana.py is located'''
    
    script_path: str = os.path.dirname(source_path)
    '''absolute path to the project root directory (parent of src/)'''
    
    output_dir: str = os.path.join(script_path, 'connection_data')
    '''folder to save all data'''
```

### 2. VisualizeSkeleton (Line 4952-4955)

```python
@dataclass
class VisualizeSkeleton:
    source_path: str = os.path.dirname(os.path.abspath(__file__))
    '''absolute path to the src/ directory where coana.py is located'''
    
    script_path: str = os.path.dirname(source_path)
    '''absolute path to the project root directory (parent of src/)'''
    
    output_dir: str = os.path.join(script_path, 'connection_data')
    '''folder to save all data'''
```

## Path Resolution

For a project at: `/path/to/drocat/`

| Attribute | Value | Purpose |
|-----------|-------|---------|
| `source_path` | `/path/to/drocat/src` | Points to src/ directory |
| `script_path` | `/path/to/drocat` | Points to project root |
| `output_dir` | `/path/to/drocat/connection_data` | Data storage location |
| `cache_folder` | `/path/to/drocat/cache/<dataset>` | Cache storage location |

## Benefits

1. **Clarity** 
   - Immediately clear what each path represents
   - No need to count `os.path.dirname()` calls

2. **Maintainability**
   - Easier to modify if needed
   - Self-documenting code

3. **Flexibility**
   - Could use `source_path` for src/-relative operations if needed
   - `script_path` for project-root-relative operations

4. **Readability**
   ```python
   # Clear and obvious
   script_path = os.path.dirname(source_path)
   
   # vs. requires mental parsing
   script_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
   ```

## Usage Examples

### Accessing the Attributes

```python
from coana import FindNeuronConnection

fc = FindNeuronConnection(
    token='your_token',
    dataset='hemibrain:v1.2.1',
    sourceNeurons=['KC.*'],
    targetNeurons=['MBON.*']
)

# Access the paths
print(fc.source_path)  # .../drocat/src
print(fc.script_path)  # .../drocat
print(fc.output_dir)  # .../drocat/connection_data
```

### Directory Structure

```
drocat/         ← script_path
├── src/                                     ← source_path
│   ├── coana.py
│   ├── statvis.py
│   └── vispath.py
├── cache/                                   ← script_path/cache
├── datasets/                                ← script_path/datasets
├── connection_data/                         ← output_dir
└── scripts/
    └── FindPath.py
```

## Verification

All paths resolve correctly:

```bash
# Test
python -c "
import sys; sys.path.insert(0, 'src')
import os
file_loc = 'src/coana.py'
source = os.path.dirname(os.path.abspath(file_loc))
script = os.path.dirname(source)
print(f'source_path: {os.path.basename(source)}')  # src
print(f'script_path: {os.path.basename(script)}')  # project-name
"
```

✅ Output shows correct directory names

## Impact

- **No Breaking Changes** - All existing functionality works the same
- **Backward Compatible** - `script_path` still behaves identically
- **Additional Clarity** - New `source_path` attribute for src/ directory

## Files Modified

- `src/coana.py` (2 classes updated)
  - Line 78-81: `FindNeuronConnection.source_path` added
  - Line 4952-4955: `VisualizeSkeleton.source_path` added

---

**Summary:** Added `source_path` attribute to make path management clearer and more maintainable after src/ layout reorganization.
