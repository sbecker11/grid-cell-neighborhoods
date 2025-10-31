// Test Runner for Grid Cell Neighborhoods
// Uses Pyodide to run Python tests in the browser

let pyodide = null;
let loading = false;
let testPages = [];
let currentPageIndex = 0;

function setStatus(className, text) {
    const statusEl = document.getElementById('status');
    const runButton = document.getElementById('runTests');
    if (statusEl) {
        statusEl.className = `status ${className}`;
        statusEl.textContent = text;
        statusEl.classList.remove('hidden');
    }
    if (runButton) {
        runButton.textContent = text;
    }
}

// Initialize Pyodide runtime (avoid shadowing global window.loadPyodide)
async function initPyodideRuntime() {
    if (pyodide) return pyodide;
    
    const outputEl = document.getElementById('output');
    const runButton = document.getElementById('runTests');
    
    setStatus('loading', 'Loading Pyodide (Python runtime)...');
    outputEl.textContent = 'Initializing Pyodide... This may take a moment on first load.\n';
    if (runButton) runButton.disabled = true;
    
    try {
        // Load Pyodide (assumes pyodide.js script is already loaded)
        if (typeof window.loadPyodide === 'undefined') {
            throw new Error('Pyodide script not loaded. Make sure pyodide.js is included in the HTML.');
        }
        pyodide = await window.loadPyodide({
            indexURL: "https://cdn.jsdelivr.net/pyodide/v0.24.1/full/",
            // Disable SharedArrayBuffer to avoid cross-origin isolation warnings
            // This is fine for single-threaded use cases like ours
            fullStdLib: true
        });
        
        setStatus('info', 'Pyodide loaded! Loading NumPy and project modules...');
        outputEl.textContent += 'Pyodide initialized successfully!\n';
        
        // Load NumPy and other required packages
        await pyodide.loadPackage(['numpy']);
        outputEl.textContent += 'NumPy loaded successfully!\n';
        
        // Load the Python modules
        setStatus('info', 'Loading project modules...');
        await loadModules();
        
        setStatus('success', '✓ Ready to run tests!');
        if (runButton) runButton.disabled = false;
        
        return pyodide;
    } catch (error) {
        setStatus('error', `Error loading Pyodide: ${error.message}`);
        outputEl.textContent += `\nError: ${error.message}\n`;
        if (runButton) runButton.disabled = true;
        throw error;
    }
}

// Load Python modules into Pyodide
async function loadModules() {
    const outputEl = document.getElementById('output');
    
    try {
        // Fetch sources
        outputEl.textContent += 'Loading grid_counting.py...\n';
        const gridCountingResponse = await fetch('grid_counting.py?v=3');
        const gridCountingCode = await gridCountingResponse.text();

        outputEl.textContent += 'Loading grid_counting_tests.py...\n';
        const testsResponse = await fetch('grid_counting_tests.py?v=14');
        const testsCode = await testsResponse.text();

        // Write to Pyodide filesystem so Python import system can find them
        pyodide.FS.writeFile('grid_counting.py', gridCountingCode);
        pyodide.FS.writeFile('grid_counting_tests.py', testsCode);

        // Import modules by name so `from grid_counting import ...` works
        // Clear module cache to ensure fresh modules
        pyodide.runPython(`
import sys, importlib
if '' not in sys.path:
    sys.path.append('')

# Clear any cached modules to prevent stale code
if 'grid_counting' in sys.modules:
    del sys.modules['grid_counting']
if 'grid_counting_tests' in sys.modules:
    del sys.modules['grid_counting_tests']

# Import fresh modules
grid_counting = importlib.import_module('grid_counting')
tests_module = importlib.import_module('grid_counting_tests')
from grid_counting_tests import run_all_tests
        `);

        outputEl.textContent += '✓ Modules imported\n';
        
    } catch (error) {
        outputEl.textContent += `Error loading modules: ${error.message}\n`;
        throw error;
    }
}

// Redirect print statements to the output area
function setupPrintCapture() {
    const outputEl = document.getElementById('output');
    
    // Create a custom print function
    pyodide.runPython(`
        import sys
        from io import StringIO
        
        class WebOutput:
            def __init__(self):
                self.buffer = []
            
            def write(self, text):
                self.buffer.append(text)
            
            def flush(self):
                pass
            
            def getvalue(self):
                return ''.join(self.buffer)
            
            def clear(self):
                self.buffer = []
        
        web_output = WebOutput()
        sys.stdout = web_output
        sys.stderr = web_output
    `);
    
    // Function to read captured output
    window.getPythonOutput = function() {
        try {
            return pyodide.runPython('web_output.getvalue()');
        } catch (e) {
            return '';
        }
    };
    
    window.clearPythonOutput = function() {
        pyodide.runPython('web_output.clear()');
    };
}

// Parse test output into individual test pages
function parseTestOutput(output) {
    const lines = output.split('\n');
    const testStartLines = [];
    let lastStart = -1000;
    for (let i = 0; i < lines.length; i++) {
        // Match both "TEST N:" (numbered) and "TEST:" (wrapper) patterns
        const trimmed = lines[i].trim();
        const numberedMatch = trimmed.match(/^TEST\s+(\d+):\s+(.+)$/);
        const unnumberedMatch = trimmed.startsWith('TEST:');
        if (numberedMatch || unnumberedMatch) {
            // Skip TEST headers that appear too close to the previous one
            // (these are usually duplicate wrapper/internal headers)
            if (i - lastStart > 6) {
                testStartLines.push(i);
                lastStart = i;
            }
        }
    }
    if (testStartLines.length === 0) {
        return [output];
    }
    const pages = [];
    for (let i = 0; i < testStartLines.length; i++) {
        let start = testStartLines[i];
        // If the line before the TEST: line is a ==== line, include it
        if (start > 0 && lines[start - 1].trim() === '='.repeat(70)) {
            start = start - 1;
        }
        const end = (i < testStartLines.length - 1) ? testStartLines[i + 1] : lines.length;
        const testLines = lines.slice(start, end);
        // Trim trailing blank lines and trailing ==== lines
        while (testLines.length > 0) {
            const lastLine = testLines[testLines.length - 1].trim();
            if (lastLine === '' || lastLine === '='.repeat(70)) {
                testLines.pop();
            } else {
                break;
            }
        }
        if (testLines.length > 0) {
            pages.push(testLines.join('\n'));
        }
    }
    console.log(`[test-runner] Detected ${pages.length} TEST sections (collapsed)`);
    return pages;
}

// Display a specific test page
function displayPage(pageIndex) {
    if (testPages.length === 0) return;
    
    if (pageIndex < 0) pageIndex = 0;
    if (pageIndex >= testPages.length) pageIndex = testPages.length - 1;
    
    currentPageIndex = pageIndex;
    const outputEl = document.getElementById('output');
    const pageInfoEl = document.getElementById('pageInfo');
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    
    // Get the test content and add test number to TEST: label
    let testContent = testPages[pageIndex];
    const testNumber = pageIndex + 1;
    
    // Replace "TEST: " with "TEST N: " where N is the test number
    testContent = testContent.replace(/^TEST: /m, `TEST ${testNumber}: `);
    
    outputEl.textContent = testContent;
    pageInfoEl.textContent = `Test ${testNumber} of ${testPages.length}`;
    
    // Update button states
    prevBtn.disabled = (pageIndex === 0);
    nextBtn.disabled = (pageIndex === testPages.length - 1);
}

// Run all tests
async function runTests() {
    if (loading) return;
    
    loading = true;
    const runButton = document.getElementById('runTests');
    const outputEl = document.getElementById('output');
    const paginationEl = document.getElementById('pagination');
    
    runButton.disabled = true;
    setStatus('loading', 'Running tests...');
    outputEl.textContent = '';
    paginationEl.classList.add('hidden');
    testPages = [];
    currentPageIndex = 0;
    
    try {
        // Load Pyodide if not already loaded
        if (!pyodide) {
            await initPyodideRuntime();
            setupPrintCapture();
        }
        
        setStatus('loading', 'Running tests...');
        
        // Clear previous output
        clearPythonOutput();
        
        // Run the test suite
        outputEl.textContent = 'Starting test execution...\n';
        outputEl.textContent += '='.repeat(70) + '\n';
        
        // Update output periodically while tests run
        const updateInterval = setInterval(() => {
            const captured = getPythonOutput();
            if (captured) {
                outputEl.textContent = captured;
            }
        }, 100);
        
        // Run tests
        try {
            pyodide.runPython(`
                # Run all tests
                run_all_tests()
            `);
            
            // Get final output
            const finalOutput = getPythonOutput();
            
            clearInterval(updateInterval);
            
            // Check if tests passed by examining output (more reliable)
            const testResult = finalOutput.includes('✓ All tests passed!') && 
                              !finalOutput.includes('✗ Test failed') &&
                              !finalOutput.includes('✗ Error:');
            
            // Parse output into pages
            testPages = parseTestOutput(finalOutput);
            
            // Display first page
            if (testPages.length > 0) {
                paginationEl.classList.remove('hidden');
                displayPage(0);
            } else {
                outputEl.textContent = finalOutput;
            }
            
            if (testResult) {
                setStatus('success', '✓ All tests passed!');
            } else {
                setStatus('error', '✗ Some tests failed');
            }
            
        } catch (error) {
            clearInterval(updateInterval);
            const errorMsg = error.toString();
            const captured = getPythonOutput();
            outputEl.textContent = (captured || '') + '\n' + errorMsg;
            
            setStatus('error', `✗ Error running tests: ${errorMsg}`);
        }
        
    } catch (error) {
        setStatus('error', `Error: ${error.message}`);
        outputEl.textContent += `\nError: ${error.message}\n`;
    } finally {
        loading = false;
        runButton.disabled = false;
    }
}

// Initialize when page loads
window.addEventListener('DOMContentLoaded', async () => {
    // Preload Pyodide in the background
    try {
        const runButton = document.getElementById('runTests');
        if (runButton) runButton.disabled = true;
        setStatus('loading', 'Loading Pyodide (Python runtime)...');
        await initPyodideRuntime();
        setupPrintCapture();
        
        const statusEl = document.getElementById('status');
        setStatus('success', 'Run Tests Now');
        if (runButton) runButton.disabled = false;
    } catch (error) {
        console.error('Failed to initialize Pyodide:', error);
    }
});

// Export for use in HTML
window.runTests = runTests;
window.showNextPage = function() { displayPage(currentPageIndex + 1); };
window.showPrevPage = function() { displayPage(currentPageIndex - 1); };
