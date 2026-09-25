# Network Blank Display - Bug Fix Summary

## Issue
Network visualization was not rendering (blank canvas) after implementing negative value handling.

## Root Causes

### 1. **mapData Function - Unevaluated Python Expressions**
**Problem**: The JavaScript `mapData()` function had Python f-string expressions that weren't being evaluated:
```javascript
'width': 'mapData(scaled_width, {min(e['data']['scaled_width'] for e in edges_data)}, ...)'
```

The `min()` and `max()` expressions were being inserted as literal strings into JavaScript, where `edges_data` doesn't exist as a Python structure.

**Fix**: Calculate min/max scaled widths in Python before generating HTML:
```python
# Calculate min and max scaled widths for mapData function
min_scaled_width = min(scaled_widths) if scaled_widths else 1
max_scaled_width = max(scaled_widths) if scaled_widths else 10
```

Then use these values in the f-string:
```javascript
'width': 'mapData(scaled_width, {min_scaled_width}, {max_scaled_width}, 1, 10)'
```

### 2. **Boolean Type Mismatch - Python vs JavaScript**
**Problem**: The `is_negative` field was using Python boolean values (`True`/`False`):
```python
'is_negative': is_negative  # Python True/False
```

This was inserted into JavaScript as:
```javascript
'is_negative': True  // Invalid JavaScript
```

The CSS selector `edge[is_negative = true]` couldn't match because:
- JavaScript expects lowercase `true`
- Python's `True` is not a valid JavaScript identifier

**Fix**: Use numeric values (0/1) instead of booleans:
```python
'is_negative': 1 if is_negative else 0  # Use 1/0 for JavaScript
```

Update selector:
```javascript
selector: 'edge[is_negative = 1]'
```

Update hover handler:
```javascript
const displayWeight = data.is_negative === 1 ? -data.weight : data.weight;
```

## Files Modified
- `src/vispath.py`:
  - Lines 2940-2944: Added min/max scaled width calculation
  - Line 2933: Changed `is_negative` from boolean to numeric (1/0)
  - Line 3568: Updated mapData in edge selector
  - Line 3597: Updated selector to `edge[is_negative = 1]`
  - Line 3606: Updated selected edge mapData
  - Line 3614: Updated highlighted edge mapData
  - Line 3787: Updated hover handler to check `=== 1`

## Testing
```bash
python -m pytest tests/core/test_audit_fixes.py -k negative
```

(The one-off `scripts/PlotPath_TestNegatives.py` harness has been removed.)

**Expected Results**:
- ✅ Network displays correctly (not blank)
- ✅ Light blue edges for negative weights
- ✅ Gray edges for positive weights
- ✅ Hover shows correct negative sign: "Weight: -75"
- ✅ Edge widths scale properly using mapData function

## Lessons Learned

1. **Type Compatibility**: Always consider JavaScript/Python type differences when generating HTML/JS from Python:
   - Booleans: Python (`True`/`False`) vs JavaScript (`true`/`false`)
   - Use numeric values (0/1) for cross-language compatibility

2. **F-String Evaluation**: Python expressions in f-strings must be evaluated BEFORE insertion into JavaScript:
   - ❌ `{min(e['data']['x'] for e in edges_data)}`  (tries to evaluate in JS)
   - ✅ `{min_value}` where `min_value` is pre-calculated in Python

3. **CSS Selectors**: Cytoscape.js selectors require exact type matching:
   - `edge[field = 1]` only matches numeric 1, not boolean true
   - `edge[field = "true"]` only matches string "true"

## Related Documentation
- [Negative Values Implementation](../technical/NEGATIVE_VALUES_IMPLEMENTATION.md)
- [Negative Values Quick Reference](../technical/NEGATIVE_VALUES_QUICKREF.md)
