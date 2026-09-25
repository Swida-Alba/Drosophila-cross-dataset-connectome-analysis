# Installation & Dependency Management Summary

> **Dated snapshot (v2.0 setup era).** This record describes the
> installation methods and dependency rationale as of its writing:
> `setup.py`, `INSTALLATION.md`, and `test_sheet_confirmation.py` no
> longer exist, extras have changed (see `pyproject.toml`), and
> `requirements.txt` now carries exact pins (converted from minimum
> versions on 2026-09-12). For the current dependency set read
> `requirements.txt` and `pyproject.toml` directly.

## Created Files

### 1. `requirements.txt` ✅
**Purpose:** Standard Python dependency list  
**Usage:** `pip install -r requirements.txt`

**Key Dependencies:**
- Core: numpy, pandas, scipy
- Visualization: plotly, matplotlib, seaborn
- Network: networkx
- Data: openpyxl, xlrd
- **GUI: PyQt5** (for fast dialogs)
- Neuroscience: neuprint-python, navis
- Optional: pytest, jupyter

### 2. `setup.py` ✅
**Purpose:** Automated installation script  
**Usage:** `python setup.py`

**Features:**
- Progress indicators for each package
- Error handling for failed installations
- PyQt5 included for fast GUI performance
- User-friendly output with emoji indicators

### 3. `pyproject.toml` ✅
**Purpose:** Modern Python project configuration  
**Usage:** `pip install .` or `pip install ".[all]"`

**Optional Dependencies:**
- `[dev]` - Development tools (pytest, jupyter)
- `[viz]` - Visualization extras (navis, trimesh, kaleido)
- `[gui]` - Alternative GUI backends (PyQt6, wxPython)
- `[all]` - Everything

### 4. `INSTALLATION.md` ✅
**Purpose:** Comprehensive installation guide  
**Contents:**
- Quick start options
- Platform-specific instructions
- Troubleshooting tips
- Verification steps

### 5. `DIALOG_PERFORMANCE_GUIDE.md` ✅
**Purpose:** GUI backend performance documentation  
**Contents:**
- Backend comparison (PyQt5 vs wxPython vs tkinter)
- Performance benchmarks
- Installation recommendations
- Technical details

## Installation Methods

### Method 1: requirements.txt (Recommended)
```bash
pip install -r requirements.txt
```
**Pros:**
- Standard Python approach
- Version pinning
- Easy to understand
- Works everywhere

### Method 2: setup.py
```bash
python setup.py
```
**Pros:**
- Progress indicators
- Error handling
- User-friendly output

### Method 3: pyproject.toml (Modern)
```bash
pip install .           # Basic install
pip install ".[dev]"    # With dev tools
pip install ".[all]"    # Everything
```
**Pros:**
- Modern Python standard
- Optional dependencies
- Better metadata

## Key Additions

### PyQt5 for Fast GUI Dialogs ⚡

**Before:**
- Slow tkinter dialogs (2-5 seconds on macOS)
- Console warnings
- Poor responsiveness

**After:**
- Instant PyQt5 dialogs (~0.1 seconds)
- Native appearance
- Smooth user experience

**Installation:**
```bash
# Included in requirements.txt
pip install PyQt5
```

### Automatic Backend Selection

The system tries backends in this order:
1. PyQt5 (fastest) ⚡⚡⚡
2. PyQt6 (also fast) ⚡⚡⚡
3. wxPython (good) ⚡⚡
4. tkinter (slow) ⚡
5. Terminal (fallback)

## Version Constraints

### Critical Constraints
- `pandas<2.0.0` - Project not yet compatible with pandas 2.x
- `numpy>=1.20.0` - For modern array features
- `PyQt5>=5.15.0` - For stable GUI support

### Recommended Versions
All packages specify minimum versions for:
- Security updates
- Bug fixes
- Feature availability

## Platform Support

### macOS ✅
- All dependencies available via pip
- PyQt5 works out of box
- tkinter included with Python

### Linux ✅
- May need system packages: `python3-tk`, `python3-dev`
- PyQt5 installable via pip or apt
- Full support for all features

### Windows ✅
- All dependencies via pip
- PyQt5 works well
- tkinter usually included

## Testing Your Installation

### Quick Test
```bash
python -c "import numpy, pandas, plotly, networkx, PyQt5; print('✅ All OK')"
```

### GUI Test
```bash
python test_sheet_confirmation.py
```

### Import Test
```python
from vispath import VisualizePath
from coana import Coana
from statvis import StatVis
```

## Troubleshooting

### Common Issues

**1. PyQt5 won't install**
```bash
pip install --upgrade pip
pip install PyQt5
```

**2. Pandas version conflict**
```bash
pip install "pandas<2"
```

**3. tkinter missing (Linux)**
```bash
sudo apt-get install python3-tk
```

**4. Import errors**
```bash
pip install --upgrade --force-reinstall -r requirements.txt
```

## Documentation Files

| File | Purpose |
|------|---------|
| `requirements.txt` | Standard dependency list |
| `setup.py` | Automated installer |
| `pyproject.toml` | Modern project config |
| `INSTALLATION.md` | Installation guide |
| `DIALOG_PERFORMANCE_GUIDE.md` | GUI performance guide |
| `README.md` | Main project documentation |

## Next Steps

### For Users
```bash
# Just install and use
pip install -r requirements.txt
python your_script.py
```

### For Developers
```bash
# Install with dev tools
pip install ".[dev]"
pytest tests/
```

### For Contributors
```bash
# Clone and install in editable mode
git clone <repo>
cd <repo>
pip install -e ".[dev]"
```

## Summary

✅ **requirements.txt** - Standard Python dependencies with PyQt5  
✅ **setup.py** - User-friendly installer with progress  
✅ **pyproject.toml** - Modern config with optional dependencies  
✅ **INSTALLATION.md** - Complete installation guide  
✅ **DIALOG_PERFORMANCE_GUIDE.md** - GUI optimization docs  

**Result:** Fast, fluent GUI dialogs with PyQt5! 🚀
