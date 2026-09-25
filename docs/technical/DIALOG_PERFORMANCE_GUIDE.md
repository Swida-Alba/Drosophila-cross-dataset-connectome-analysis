# GUI Dialog Performance Optimization

## Overview

The sheet selection dialog supports **multiple GUI backends** with
automatic fallback; the file picker uses the same GUI cascade but, when
NO GUI library is available, it fails with an error (plus a pip-install
hint) rather than falling back to the terminal — terminal selection is
sheet-selection only:

1. **PyQt5** (Recommended - Fastest ⚡)
2. **PyQt6** (Also Fast ⚡)
3. **wxPython** (Good Performance 👍)
4. **tkinter** (Slowest GUI, but always available)
5. **Terminal** (Sheet selection only — no GUI library needed)

## Performance Comparison

| Backend | Speed | Native Look | Installation |
|---------|-------|-------------|--------------|
| **PyQt5** | ⚡⚡⚡ Very Fast | ✅ Excellent | `pip install PyQt5` |
| **PyQt6** | ⚡⚡⚡ Very Fast | ✅ Excellent | `pip install PyQt6` |
| **wxPython** | ⚡⚡ Fast | ✅ Good | `pip install wxPython` |
| **tkinter** | ⚡ Slow | ⚠️ Basic | Built-in (usually) |
| **Terminal** | N/A (text input) | ❌ Text only | No dependencies |

## Installation

### Option 1: PyQt5 (Recommended)

```bash
pip install PyQt5
```

**Benefits:**
- Lightning-fast dialogs
- Native macOS/Windows/Linux appearance
- Most responsive UI
- Best for interactive workflows

### Option 2: PyQt6

```bash
pip install PyQt6
```

**Benefits:**
- Same performance as PyQt5
- Modern Qt6 framework
- Good for newer systems

### Option 3: wxPython

```bash
pip install wxPython
```

**Benefits:**
- Fast performance
- Native widgets
- Good alternative to PyQt

### Option 4: tkinter (Default)

Usually pre-installed with Python. If not:

**macOS:** Already included  
**Ubuntu/Debian:**
```bash
sudo apt-get install python3-tk
```

**Windows:** Reinstall Python with tkinter option enabled

## Usage

The system **automatically** selects the best available backend. No configuration needed!

```python
from vispath_pkg import VisualizePath

# File picker will use fastest available backend
vis = VisualizePath(
    path_file=None,  # Opens file picker
    output_folder='./output'
)
```

## Backend Selection Order

When you trigger a file picker or sheet selection dialog:

1. **Try PyQt5** - If installed, use it (fastest)
2. **Try PyQt6** - If PyQt5 not available
3. **Try wxPython** - If no PyQt available
4. **Try tkinter** - Last GUI option
5. **Sheet selection only**: fall back to the terminal prompt; the FILE
   picker instead raises an error listing the pip-install options

## Performance Tips

### For Best Performance:

```bash
# Install PyQt5 for fastest dialogs
pip install PyQt5

# Or if using Python 3.10+
pip install PyQt6
```

### For Minimal Installation:

Use built-in tkinter (slower but no extra dependencies)

### For Headless Servers:

Terminal input works automatically when no GUI available

## Example Speed Comparison

**Opening file picker dialog:**

| Backend | Time | Notes |
|---------|------|-------|
| PyQt5 | ~0.1s | Instant, native |
| PyQt6 | ~0.1s | Instant, native |
| wxPython | ~0.2s | Very quick |
| tkinter | ~2-5s | Noticeable delay on macOS |
| Terminal | N/A | Text-based |

**Sheet selection dialog:**

| Backend | Time | Notes |
|---------|------|-------|
| PyQt5 | ~0.05s | Instant |
| PyQt6 | ~0.05s | Instant |
| wxPython | ~0.1s | Quick |
| tkinter | ~1-3s | Slow on macOS |
| Terminal | N/A | Text-based |

## Troubleshooting

### "No GUI library available" Error

Install at least one GUI library:

```bash
pip install PyQt5  # Recommended
```

### PyQt5 Import Error

Try alternative:

```bash
pip install PyQt6
# or
pip install wxPython
```

### macOS tkinter Warning

If you see warnings like:
```
The class 'NSOpenPanel' overrides the method identifier...
```

This is a tkinter issue on macOS. **Solution:** Install PyQt5 for cleaner, faster dialogs:

```bash
pip install PyQt5
```

### Dialogs Don't Appear

If dialogs don't show up:

1. **Check GUI backend:** the backend name appears in the error path
   when a library fails to import (there is no success-path backend
   banner, and no environment variable forces terminal mode)
2. **Reinstall GUI library:**
   ```bash
   pip uninstall PyQt5
   pip install PyQt5
   ```

## Technical Details

### Architecture

The implementation uses a cascading try-except pattern:

```python
# Try backends in order of performance
result = self._prompt_sheet_pyqt5(...)  # Fastest
if result is False:
    result = self._prompt_sheet_pyqt6(...)
if result is False:
    result = self._prompt_sheet_wx(...)
if result is False:
    result = self._prompt_sheet_tkinter(...)
if result is False:
    result = self._prompt_sheet_terminal(...)  # Fallback
```

### File Picker Backends

Each backend implements:
- Native file dialog
- File type filtering (.xlsx, .xls, .csv)
- Initial directory setting
- Cross-platform path normalization

### Sheet Selection Backends

Each backend provides:
- List of all sheets with metadata (rows, cols)
- Pre-selection of auto-detected sheet
- Visual highlighting of suggested sheet
- Double-click and keyboard support
- OK/Cancel buttons

## Recommendations

### For Development Workflow

```bash
pip install PyQt5
```

You'll get:
- ⚡ Instant file dialogs
- 🎨 Native macOS/Windows appearance
- 🚀 Smooth user experience
- ✨ No tkinter warnings

### For Production/Deployment

Include in `requirements.txt`:

```txt
# Recommended for best performance
PyQt5>=5.15.0

# Or alternative
# PyQt6>=6.0.0
# wxPython>=4.2.0
```

### For Minimal Installation

Just use built-in tkinter (no extra install needed, but slower)

## Conclusion

- **Best Experience:** Install PyQt5
- **Good Alternative:** Install wxPython
- **Minimal Setup:** Use built-in tkinter
- **Automatic Fallback:** System picks best available
- **No Configuration:** Works out of the box!

---

**Quick Start:**

```bash
# One command for best performance
pip install PyQt5

# Then use normally
python your_script.py
```

Enjoy lightning-fast dialogs! ⚡
