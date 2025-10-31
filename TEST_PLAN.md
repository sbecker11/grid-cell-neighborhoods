# Test Plan

This document outlines the complete testing procedure for the grid cell neighborhoods project.

## Prerequisites

Before running tests, ensure:
- Python 3.11 is installed
- Virtual environment is set up (recommended) or use system Python
- Dependencies installed via `requirements.txt`

**Note:** Setup steps are provided in each OS-specific section below. Follow the setup instructions for your operating system.

---

## Part 1: Local macOS Shell Tests

Local macOS shell tests run Python unit tests directly in your terminal on macOS.

### 1.1 macOS Shell Tests

#### 1.1.0 Setup Steps (macOS)

1. **Upgrade pip** (recommended for better dependency resolution):
   ```bash
   python3 -m pip install --upgrade "pip>=24.0"
   ```

2. **Create and activate virtual environment** (recommended):
   ```bash
   # Create venv (if it doesn't exist)
   python3 -m venv venv
   
   # Activate venv
   source venv/bin/activate
   ```

3. **Install dependencies from requirements.txt**:
   ```bash
   pip install -r requirements.txt
   ```
   
   This installs:
   - NumPy (required)
   - PyTorch, JAX, JAXlib (optional, for hardware detection tests)

#### Test 1.1.1: Python Unit Tests (macOS)

**Prerequisite:** Complete setup steps above, then activate venv:
```bash
source venv/bin/activate
```

**Command:**
```bash
python grid_counting_tests.py
```

**Expected Results:**
- All 29 tests should pass
- Output should show numbered test headers: `TEST 1:`, `TEST 2:`, etc.
- Each numbered header wrapped with `====` lines above and below
- Operating system information: `Operating System: Darwin ...`
- Platform: `arm64` (Apple Silicon) or `x86_64` (Intel)
- Hardware detection shows MPS availability (if Apple Silicon) or no GPU
- Final message: `✓ All tests passed!`

**Verification Checklist:**
- [ ] Test count: 31 tests total
- [ ] Test headers are numbered (TEST 1, TEST 2, ..., TEST 29)
- [ ] Each numbered header has `====` lines above and below
- [ ] No duplicate "TEST:" headers in output
- [ ] OS shows "Darwin" in first test output
- [ ] Hardware detection shows macOS-specific accelerators (MPS if Apple Silicon)
- [ ] Exit code: 0 (success)

---

## Part 2: Local Windows Shell Tests

Local Windows shell tests run Python unit tests directly in your terminal on Windows.

### 2.1 Windows Shell Tests

#### 2.1.0 Setup Steps (Windows)

1. **Upgrade pip** (recommended for better dependency resolution):
   
   In PowerShell or Command Prompt:
   ```bash
   python -m pip install --upgrade "pip>=24.0"
   ```

2. **Create and activate virtual environment** (recommended):
   
   In PowerShell:
   ```bash
   # Create venv (if it doesn't exist)
   python -m venv venv
   
   # Activate venv (PowerShell)
   venv\Scripts\Activate.ps1
   ```
   
   In Command Prompt:
   ```bash
   # Create venv (if it doesn't exist)
   python -m venv venv
   
   # Activate venv (Command Prompt)
   venv\Scripts\activate.bat
   ```
   
   **Note:** If PowerShell execution policy prevents activation, run:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

3. **Install dependencies from requirements.txt**:
   ```bash
   pip install -r requirements.txt
   ```
   
   This installs:
   - NumPy (required)
   - PyTorch, JAX, JAXlib (optional, for hardware detection tests)

#### Test 2.1.1: Python Unit Tests (Windows)

**Prerequisite:** Complete setup steps above, then activate venv:

In PowerShell:
```bash
venv\Scripts\Activate.ps1
```

In Command Prompt:
```bash
venv\Scripts\activate.bat
```

**Command:**
```bash
python grid_counting_tests.py
```

**Expected Results:**
- All 29 tests should pass
- Output should show numbered test headers: `TEST 1:`, `TEST 2:`, etc.
- Each numbered header wrapped with `====` lines above and below
- Operating system information: `Operating System: Windows ...`
- Platform: `AMD64`
- Hardware detection shows CUDA availability (if NVIDIA GPU) or no GPU
- Final message: `✓ All tests passed!`

**Verification Checklist:**
- [ ] Test count: 31 tests total
- [ ] Test headers are numbered (TEST 1, TEST 2, ..., TEST 29)
- [ ] Each numbered header has `====` lines above and below
- [ ] No duplicate "TEST:" headers in output
- [ ] OS shows "Windows" in first test output
- [ ] Hardware detection shows Windows-specific accelerators (CUDA if NVIDIA GPU)
- [ ] Exit code: 0 (success)

---

## Part 3: Local Linux Shell Tests

Local Linux shell tests run Python unit tests directly in your terminal on Linux.

### 3.1 Linux Shell Tests

#### 3.1.0 Setup Steps (Linux)

1. **Upgrade pip** (recommended for better dependency resolution):
   ```bash
   python3 -m pip install --upgrade "pip>=24.0"
   ```

2. **Create and activate virtual environment** (recommended):
   ```bash
   # Create venv (if it doesn't exist)
   python3 -m venv venv
   
   # Activate venv
   source venv/bin/activate
   ```

3. **Install dependencies from requirements.txt**:
   ```bash
   pip install -r requirements.txt
   ```
   
   This installs:
   - NumPy (required)
   - PyTorch, JAX, JAXlib (optional, for hardware detection tests)

#### Test 3.1.1: Python Unit Tests (Linux)

**Prerequisite:** Complete setup steps above, then activate venv:
```bash
source venv/bin/activate
```

**Command:**
```bash
python grid_counting_tests.py
```

**Expected Results:**
- All 29 tests should pass
- Output should show numbered test headers: `TEST 1:`, `TEST 2:`, etc.
- Each numbered header wrapped with `====` lines above and below
- Operating system information: `Operating System: Linux ...`
- Platform: `x86_64` (most common)
- Hardware detection shows CUDA availability (if NVIDIA GPU) or no GPU
- Final message: `✓ All tests passed!`

**Verification Checklist:**
- [ ] Test count: 31 tests total
- [ ] Test headers are numbered (TEST 1, TEST 2, ..., TEST 29)
- [ ] Each numbered header has `====` lines above and below
- [ ] No duplicate "TEST:" headers in output
- [ ] OS shows "Linux" in first test output
- [ ] Hardware detection shows Linux-specific accelerators (CUDA if NVIDIA GPU)
- [ ] Exit code: 0 (success)

---

## Part 4: Local Web Tests

Local web tests run the test suite in your browser using Pyodide. Choose your operating system below.

### 4.1 macOS Web Tests

#### 4.1.0 Prerequisites (macOS)

Complete Part 1 setup steps (1.1.0) before running web tests. The server requires dependencies to be installed.

#### Test 4.1.1: Browser-Based Tests (macOS)

**Step 1: Start the web server**
```bash
./start_web_tester.sh
```

**Expected Results:**
- Script checks port 8000
- If port 8000 is in use, prompts: `Kill these and use port 8000? [y/N]:`
- After confirmation, server starts
- Displays URL: `http://localhost:8000` (or next available port)
- Server runs until Ctrl+C

**Verification Checklist:**
- [ ] Port 8000 check works correctly
- [ ] Kill prompt appears if port is occupied
- [ ] Server URL is displayed clearly
- [ ] Server starts without errors

**Step 2: Open browser and run tests**

1. Open the displayed URL in your browser (usually `http://localhost:8000`)
2. Wait for Pyodide to load (may take 10-30 seconds)
3. Click "Run Tests Now" button
4. Navigate through test results using pagination

**Expected Results:**
- Pyodide loads successfully (no console errors)
- "Run Tests Now" button is enabled after loading
- All 31 tests run and pass
- Pagination shows "Test 1 of 29", "Test 2 of 29", etc.
- Each test page shows:
  - Numbered test header
  - Grid visualizations (where applicable)
  - Test output (ASCII art, counts, etc.)
  - ✓ Pass indicators
- No duplicate headers
- No `FS.genericErrors[44]` or similar Pyodide errors

**Verification Checklist:**
- [ ] Pyodide loads without errors
- [ ] Button becomes enabled after loading
- [ ] Test count: 31 tests
- [ ] Pagination works (next/previous buttons)
- [ ] Each test shows correct output
- [ ] No console errors in browser DevTools
- [ ] Tests match local Python test output format

**Step 3: Stop the server**
- Press `Ctrl+C` in terminal
- Server shuts down cleanly

---

### 4.2 Windows Web Tests

#### 4.2.0 Prerequisites (Windows)

Complete Part 2 setup steps (2.1.0) before running web tests. The server requires dependencies to be installed.

#### Test 4.2.1: Browser-Based Tests (Windows)

**Step 1: Start the web server**

In PowerShell or Command Prompt:
```bash
.\start_web_tester.sh
```

**Note:** If using Git Bash on Windows, the bash script should work. Otherwise, you can run directly:
```bash
python web_tester\server.py
```

**Expected Results:**
- Script checks port 8000
- If port 8000 is in use, prompts: `Kill these and use port 8000? [y/N]:`
- After confirmation, server starts
- Displays URL: `http://localhost:8000` (or next available port)
- Server runs until Ctrl+C

**Verification Checklist:**
- [ ] Port 8000 check works correctly
- [ ] Kill prompt appears if port is occupied
- [ ] Server URL is displayed clearly
- [ ] Server starts without errors

**Step 2: Open browser and run tests**

1. Open the displayed URL in your browser (usually `http://localhost:8000`)
2. Wait for Pyodide to load (may take 10-30 seconds)
3. Click "Run Tests Now" button
4. Navigate through test results using pagination

**Expected Results:** (Same as macOS browser tests)
- Pyodide loads successfully (no console errors)
- "Run Tests Now" button is enabled after loading
- All 31 tests run and pass
- Pagination shows "Test 1 of 29", "Test 2 of 29", etc.

**Verification Checklist:**
- [ ] Pyodide loads without errors
- [ ] Button becomes enabled after loading
- [ ] Test count: 31 tests
- [ ] Pagination works (next/previous buttons)
- [ ] Each test shows correct output
- [ ] No console errors in browser DevTools

**Step 3: Stop the server**
- Press `Ctrl+C` in terminal
- Server shuts down cleanly

---

### 4.3 Linux Web Tests

#### 4.3.0 Prerequisites (Linux)

Complete Part 3 setup steps (3.1.0) before running web tests. The server requires dependencies to be installed.

#### Test 4.3.1: Browser-Based Tests (Linux)

**Step 1: Start the web server**
```bash
./start_web_tester.sh
```

**Expected Results:**
- Script checks port 8000
- If port 8000 is in use, prompts: `Kill these and use port 8000? [y/N]:`
- After confirmation, server starts
- Displays URL: `http://localhost:8000` (or next available port)
- Server runs until Ctrl+C

**Verification Checklist:**
- [ ] Port 8000 check works correctly
- [ ] Kill prompt appears if port is occupied
- [ ] Server URL is displayed clearly
- [ ] Server starts without errors

**Step 2: Open browser and run tests**

1. Open the displayed URL in your browser (usually `http://localhost:8000`)
2. Wait for Pyodide to load (may take 10-30 seconds)
3. Click "Run Tests Now" button
4. Navigate through test results using pagination

**Expected Results:** (Same as macOS/Windows browser tests)
- Pyodide loads successfully (no console errors)
- "Run Tests Now" button is enabled after loading
- All 31 tests run and pass
- Pagination shows "Test 1 of 29", "Test 2 of 29", etc.

**Verification Checklist:**
- [ ] Pyodide loads without errors
- [ ] Button becomes enabled after loading
- [ ] Test count: 31 tests
- [ ] Pagination works (next/previous buttons)
- [ ] Each test shows correct output
- [ ] No console errors in browser DevTools

**Step 3: Stop the server**
- Press `Ctrl+C` in terminal
- Server shuts down cleanly

---

## Part 5: GitHub Pages Tests

GitHub Pages tests verify that the web tester works when deployed as a static site on GitHub Pages.

### Test 5.1: Enable GitHub Pages

**Prerequisites:**
- Repository must be public (or GitHub Pages enabled for private repos)
- Push changes to main branch

**Steps:**

1. Go to your GitHub repository
2. Click **Settings** tab
3. Navigate to **Pages** in the left sidebar
4. Under **Source**, select:
   - **Deploy from a branch**
   - Branch: `main`
   - Folder: `/web_tester` (or root `/` if files are at root)
   - Click **Save**

**Expected Results:**
- GitHub Pages site is enabled
- URL displayed: `https://<username>.github.io/<repository-name>/`
- Site deploys automatically on push to main branch

**Verification Checklist:**
- [ ] Pages setting saved successfully
- [ ] Deployment URL is shown
- [ ] Initial deployment completes (may take 1-2 minutes)

---

### Test 5.2: Verify GitHub Pages Deployment

**Steps:**

1. Wait for deployment to complete (check Actions tab)
2. Visit the GitHub Pages URL
3. Verify the test runner page loads

**Expected Results:**
- Page loads without errors
- Pyodide script loads successfully
- Test runner UI is visible
- "Run Tests Now" button is present (initially disabled)

**Verification Checklist:**
- [ ] Page loads successfully
- [ ] No 404 errors
- [ ] Pyodide CDN loads (check Network tab)
- [ ] UI elements are visible
- [ ] No console errors in browser DevTools

---

### Test 5.3: Run Tests on GitHub Pages

**Steps:**

1. Open the GitHub Pages URL in your browser
2. Wait for Pyodide to load (may take 10-30 seconds)
3. Click "Run Tests Now" button
4. Navigate through test results using pagination

**Expected Results:**
- Pyodide loads successfully (no console errors)
- "Run Tests Now" button becomes enabled after loading
- All 31 tests run and pass
- Pagination shows "Test 1 of 29", "Test 2 of 29", etc.
- Each test page shows:
  - Numbered test header
  - Grid visualizations (where applicable)
  - Test output (ASCII art, counts, etc.)
  - ✓ Pass indicators
- No duplicate headers
- No `FS.genericErrors[44]` or similar Pyodide errors

**Verification Checklist:**
- [ ] Pyodide loads without errors
- [ ] Button becomes enabled after loading
- [ ] Test count: 31 tests
- [ ] Pagination works (next/previous buttons)
- [ ] Each test shows correct output
- [ ] No console errors in browser DevTools
- [ ] Tests match local web test output format
- [ ] All 29 tests pass

---

### Test 5.4: Test Cross-Platform Access

**Steps:**

Test the GitHub Pages site from different environments:
1. Open the GitHub Pages URL on different devices/browsers
2. Test on mobile device (if applicable)
3. Test from different network locations

**Expected Results:**
- Site loads correctly on all tested browsers
- Pyodide loads successfully on all browsers
- Tests run successfully across different environments
- No CORS or loading errors

**Verification Checklist:**
- [ ] Works on Chrome/Edge (Chromium-based)
- [ ] Works on Firefox
- [ ] Works on Safari (if tested)
- [ ] Mobile browsers (if applicable)
- [ ] No CORS errors in console
- [ ] CDN resources load successfully

---

## Part 6: GitHub Actions Tests (Automated Cross-Platform Testing)

GitHub Actions automatically runs tests on all three operating systems (macOS, Windows, Linux) whenever code is pushed or a pull request is created.

### Test 6.1: Trigger GitHub Actions Workflow

**Prerequisites:**
- GitHub Actions enabled (default for public repos)

**Steps - Option A: Manual Trigger (No push/PR needed):**

1. Go to your GitHub repository
2. Click **"Actions"** tab at the top
3. Select **"Test Suite"** workflow from the left sidebar
4. Click **"Run workflow"** button (top right)
5. Select branch (usually `main`)
6. Click **"Run workflow"** button

**Steps - Option B: Push to trigger automatically:**

1. **Push changes to trigger workflow:**
   ```bash
   git add .
   git commit -m "Your commit message"
   git push origin main
   ```

2. **Workflow automatically starts** after push

**Steps - Option C: Create Pull Request:**

1. ```bash
   git checkout -b feature-branch
   git add .
   git commit -m "Your changes"
   git push origin feature-branch
   ```
2. Create PR via GitHub web interface
3. **Workflow automatically starts** after PR creation

**Expected Results:**
- GitHub Actions workflow triggers automatically
- Workflow runs 3 jobs sequentially (not in parallel):
  1. Test on macOS (Python 3.11)
  2. Test on Windows (Python 3.11)
  3. Test on Ubuntu/Linux (Python 3.11)

**Verification Checklist:**
- [ ] Workflow appears in GitHub Actions tab
- [ ] All 3 OS jobs appear in the workflow run
- [ ] Jobs run sequentially (one completes before next starts)

---

### Test 6.2: Monitor GitHub Actions Execution

**Steps:**

1. Go to your GitHub repository
2. Click **"Actions"** tab at the top
3. Select the latest workflow run
4. Monitor job execution

**Expected Behavior:**
- Jobs run **sequentially** (not in parallel):
  - macOS job completes first
  - Windows job starts after macOS completes
  - Ubuntu job starts after Windows completes
- Each job shows progress:
  - Checkout code
  - Set up Python 3.11
  - Cache pip packages
  - Install dependencies (NumPy, PyTorch, JAX)
  - Run grid counting tests (`python grid_counting_tests.py`)
  - All 31 tests pass

**Verification Checklist:**
- [ ] macOS job completes successfully
- [ ] Windows job completes successfully
- [ ] Ubuntu job completes successfully
- [ ] Each job shows "✓ All tests passed!" in output
- [ ] All jobs show green checkmarks (✓)
- [ ] Total execution time displayed

---

### Test 6.3: Review Test Output for Each OS

For each OS job, verify the test output shows correct platform information:

**macOS Job:**
- Operating System: `Darwin ...`
- Platform: `arm64` (Apple Silicon) or `x86_64` (Intel)
- Hardware detection mentions MPS (if Apple Silicon) or no GPU

**Windows Job:**
- Operating System: `Windows ...`
- Platform: `AMD64`
- Hardware detection mentions CUDA (if available) or no GPU

**Linux/Ubuntu Job:**
- Operating System: `Linux ...`
- Platform: `x86_64`
- Hardware detection mentions CUDA (if available) or no GPU

**Verification Checklist:**
- [ ] Each OS shows correct platform information in first test
- [ ] Test output format matches local Python tests
- [ ] Numbered test headers (TEST 1 through TEST 31)
- [ ] All 31 tests pass on all 3 OSes
- [ ] No errors or warnings in job logs
- [ ] Exit code: 0 (success) for all jobs

---

### Test 6.4: Verify Cross-Platform Consistency

**Steps:**

1. Compare test outputs across all 3 OSes
2. Verify same tests pass on all platforms
3. Check that OS-specific information differs (as expected)

**Expected Results:**
- All 31 tests pass on all 3 operating systems
- Test logic results are consistent across platforms
- Only OS/platform information differs (expected)
- Output formatting is consistent (numbered headers, spacing)

**Verification Checklist:**
- [ ] Same test count (31) on all OSes
- [ ] Same test names and order on all OSes
- [ ] Same assertion results (tests behave identically)
- [ ] Consistent output formatting across all OSes
- [ ] OS information correctly reflects each platform

---

### Test 6.5: Handle Test Failures

If a test fails on any OS:

**Steps:**

1. Click on the failed job in GitHub Actions
2. Expand the "Run grid counting tests" step
3. Review error messages and stack traces
4. Check which test(s) failed
5. Compare with local test results

**Common Issues:**

- **Dependency installation failures**: Check `requirements.txt` is correct
- **Python version mismatch**: Ensure Python 3.11 is specified
- **Path issues**: Check file paths work on Windows (backslashes) and Unix (forward slashes)
- **OS-specific behavior**: Some tests may behave differently (e.g., hardware detection)

**Verification Checklist:**
- [ ] Understand why test failed
- [ ] Reproduce failure locally if possible
- [ ] Fix issue and push updated code
- [ ] Verify fix by checking new workflow run

---

### Test 6.6: View Workflow History

**Steps:**

1. Go to repository → **Actions** tab
2. View list of all workflow runs
3. Click any run to see details

**Information Available:**
- Success/failure status for each run
- Execution time for each job
- Which commit triggered the run
- Branch or PR that triggered the run

**Verification Checklist:**
- [ ] Can see workflow run history
- [ ] Can identify which commit triggered each run
- [ ] Can view detailed logs for any run
- [ ] Can see job execution times

---

## Test Summary Checklist

### Part 1: Local macOS Shell Tests
- [ ] Python unit tests pass (31/31)
- [ ] Output formatting is correct
- [ ] No errors in console/logs
- [ ] OS information displayed correctly

### Part 2: Local Windows Shell Tests
- [ ] Python unit tests pass (31/31)
- [ ] Output formatting is correct
- [ ] No errors in console/logs
- [ ] OS information displayed correctly

### Part 3: Local Linux Shell Tests
- [ ] Python unit tests pass (31/31)
- [ ] Output formatting is correct
- [ ] No errors in console/logs
- [ ] OS information displayed correctly

### Part 4: Local Web Tests
- [ ] Browser-based tests pass (31/31)
- [ ] Pyodide loads successfully
- [ ] Pagination works correctly
- [ ] No console errors in browser

### Part 5: GitHub Pages Tests
- [ ] GitHub Pages enabled
- [ ] Site deploys successfully
- [ ] Page loads without errors
- [ ] All 31 tests pass on GitHub Pages
- [ ] Works across different browsers
- [ ] No CORS or loading errors

### Part 6: GitHub Actions Tests
- [ ] Workflow triggers on push/PR
- [ ] All 3 OS jobs run successfully (macOS, Windows, Linux)
- [ ] All 31 tests pass on macOS
- [ ] All 31 tests pass on Windows
- [ ] All 31 tests pass on Ubuntu/Linux
- [ ] Jobs run sequentially (one at a time)
- [ ] OS-specific information displayed correctly
- [ ] Output formatting consistent across all OSes

---

## Troubleshooting

### Local Python Tests Fail
- **Check Python version**: `python3 --version` (should be 3.11+)
- **Upgrade pip first**: `python -m pip install --upgrade "pip>=24.0"`
- **Install dependencies**: `pip install -r requirements.txt`
- **If using venv, activate it**: `source venv/bin/activate` (macOS/Linux)
- **Check import path**: `python3 -c "from grid_counting import DenseGrid"`
- **Verify NumPy installed**: `python3 -c "import numpy; print(numpy.__version__)"`

### Browser Tests Fail
- Clear browser cache (hard refresh: Cmd+Shift+R)
- Check browser console for errors
- Verify server is running: `curl http://localhost:8000`
- Try different port if 8000 is occupied

### GitHub Actions Fail
- Check workflow file syntax: `.github/workflows/test.yml`
- Verify file paths match repository structure
- Check if dependencies install correctly
- Review job logs for specific error messages
- Ensure Python 3.11 is available on GitHub runners (should be by default)

### Tests Run but Output Format Wrong
- Verify `grid_counting_tests.py` has correct formatting code
- Check `web_tester/client.js` parsing logic
- Clear browser cache and restart server

---

## Success Criteria

All tests pass when:
1. ✅ Part 1: Local macOS shell tests: 31/31 passing
2. ✅ Part 2: Local Windows shell tests: 31/31 passing
3. ✅ Part 3: Local Linux shell tests: 31/31 passing
4. ✅ Part 4: Local web tests: 31/31 passing
5. ✅ Part 5: GitHub Pages: Site deploys and all 31 tests pass
6. ✅ Part 6: GitHub Actions: All 3 OS jobs pass with 31/31 tests each
7. ✅ Output formatting consistent across all test environments
8. ✅ No errors in logs or console

---

## Quick Reference Commands

```bash
# Setup (one-time)
python -m pip install --upgrade "pip>=24.0"
pip install -r requirements.txt

# If using virtual environment
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Local Python tests
python grid_counting_tests.py  # (with venv activated)
# OR
python3 grid_counting_tests.py  # (system Python)

# Start web server
./start_web_tester.sh

# Check GitHub Actions status
# (via GitHub web interface)

# Verify test file structure
ls -1 grid_counting*.py start_web_tester.sh

# Check Python version
python3 --version

# Verify dependencies installed
python3 -c "import numpy; print('NumPy:', numpy.__version__)"
```

