# GitHub Pages Test Runner

This directory contains the static website for running unit tests in the browser using Pyodide.

## Files

- `index.html` - Main HTML page with UI
- `test_runner.js` - JavaScript that loads Pyodide and executes tests
- `grid_counting.py` - Main Python module
- `grid_counting_tests.py` - Test suite

## Setup for GitHub Pages

The site will be automatically deployed to GitHub Pages when you:
1. Enable GitHub Pages in repository settings (Settings → Pages)
2. Select source: "GitHub Actions" (uses `.github/workflows/pages.yml`)
3. Push to main branch

The site will be available at: `https://<username>.github.io/<repository-name>/`

## Local Testing

To test locally before deploying:

1. Start a local web server (Python 3):
   ```bash
   cd docs
   python3 -m http.server 8000
   ```

2. Open in browser:
   ```
   http://localhost:8000
   ```

## How It Works

1. **Pyodide** - Python runtime compiled to WebAssembly runs entirely in the browser
2. **No server required** - All code execution happens client-side
3. **NumPy support** - Pyodide includes NumPy, which is used by the test suite
4. **Live output** - Test results are captured and displayed in real-time

## Notes

- First load may take 10-20 seconds to download and initialize Pyodide (~8MB)
- Subsequent loads are faster due to browser caching
- JAX/PyTorch GPU acceleration won't be available in browser (NumPy fallback will be used)
- All tests that don't require GPU-specific features should pass

