# Web Tester

This directory contains the web-based test runner for running unit tests in the browser using Pyodide.

## Files

- `index.html` - Main HTML page with UI
- `client.js` - Client-side JavaScript that loads Pyodide and executes tests in browser
- `server.py` - Server-side Python script to serve files locally for development
- `tests.js` - Utility script for testing output parsing
- `favicon.ico` - Favicon for the web interface

## Setup for GitHub Pages

The site will be automatically deployed to GitHub Pages when you:
1. Enable GitHub Pages in repository settings (Settings → Pages)
2. Select source: "GitHub Actions" (uses `.github/workflows/pages.yml`)
3. Push to main branch

The site will be available at: `https://<username>.github.io/<repository-name>/`

## Local Testing

To test locally before deploying:

1. Start the local web server:
   ```bash
   python3 web_tester/server.py
   ```

2. The server will automatically:
   - Find an available port (starting at 8000)
   - Serve files from the `web_tester/` directory
   - Display the URL to open (e.g., `http://localhost:8000`)

3. Open the URL in your browser and click "Run Tests Now"

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

