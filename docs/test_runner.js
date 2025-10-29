// Test Runner for Grid Cell Neighborhoods
// Uses Pyodide to run Python tests in the browser

let pyodide = null;
let loading = false;
let testPages = [];
let currentPageIndex = 0;

// Initialize Pyodide runtime (avoid shadowing global window.loadPyodide)
async function initPyodideRuntime() {
    if (pyodide) return pyodide;
    
    const statusEl = document.getElementById('status');
    const outputEl = document.getElementById('output');
    
    statusEl.className = 'status loading';
    statusEl.textContent = 'Loading Pyodide (Python runtime)...';
    statusEl.classList.remove('hidden');
    outputEl.textContent = 'Initializing Pyodide... This may take a moment on first load.\n';
    
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
        
        statusEl.className = 'status info';
        statusEl.textContent = 'Pyodide loaded! Loading NumPy and project modules...';
        outputEl.textContent += 'Pyodide initialized successfully!\n';
        
        // Load NumPy and other required packages
        await pyodide.loadPackage(['numpy']);
        outputEl.textContent += 'NumPy loaded successfully!\n';
        
        // Load the Python modules
        await loadModules();
        
        statusEl.className = 'status success';
        statusEl.textContent = '✓ Ready to run tests!';
        
        return pyodide;
    } catch (error) {
        statusEl.className = 'status error';
        statusEl.textContent = `Error loading Pyodide: ${error.message}`;
        outputEl.textContent += `\nError: ${error.message}\n`;
        throw error;
    }
}

// Load Python modules into Pyodide
async function loadModules() {
    const outputEl = document.getElementById('output');
    
    try {
        // Fetch sources
        outputEl.textContent += 'Loading grid_counting.py...\n';
        const gridCountingResponse = await fetch('grid_counting.py');
        const gridCountingCode = await gridCountingResponse.text();

        outputEl.textContent += 'Loading grid_counting_tests.py...\n';
        const testsResponse = await fetch('grid_counting_tests.py');
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
    const pages = [];
    let currentPage = [];
    let testStartIndex = -1;
    
    // Find all test boundaries (lines with ===...=== that are followed by TEST:)
    const testBoundaries = [];
    for (let i = 0; i < lines.length - 1; i++) {
        const line = lines[i];
        const nextLine = lines[i + 1];
        // Look for separator line followed by TEST: in next line
        if (line.trim().match(/^={20,}/) && nextLine.trim().startsWith('TEST:')) {
            testBoundaries.push(i);
        }
    }
    
    // If no test boundaries found, return whole output as one page
    if (testBoundaries.length === 0) {
        return [output];
    }
    
    // Process each test - include everything from boundary to next boundary
    for (let i = 0; i < testBoundaries.length; i++) {
        const startIdx = testBoundaries[i];
        const endIdx = (i < testBoundaries.length - 1) ? testBoundaries[i + 1] : lines.length;
        
        // Extract test content (from separator through next boundary)
        const testLines = lines.slice(startIdx, endIdx);
        
        // Remove trailing blank lines to clean up the output
        while (testLines.length > 0 && testLines[testLines.length - 1].trim() === '') {
            testLines.pop();
        }
        
        if (testLines.length > 0) {
            pages.push(testLines.join('\n'));
        }
    }
    
    // Add any trailing content after last test
    if (testBoundaries.length > 0) {
        const lastBoundary = testBoundaries[testBoundaries.length - 1];
        const lastTestEnd = lines.length;
        if (lastTestEnd > lastBoundary) {
            // Check if there's additional summary content
            const remainingLines = lines.slice(lastBoundary);
            let hasSummary = false;
            for (const line of remainingLines) {
                if (line.includes('All tests') || line.includes('tests passed') || 
                    line.includes('some tests failed')) {
                    hasSummary = true;
                    break;
                }
            }
            if (hasSummary) {
                // Extract from end of last test
                const lastTestPage = pages[pages.length - 1];
                const lastTestLines = lastTestPage.split('\n');
                const summaryStart = lastTestLines.findIndex(l => 
                    l.includes('All tests') || l.includes('tests passed') || 
                    l.includes('some tests failed'));
                if (summaryStart === -1) {
                    // Summary not in last test, add as new page
                    const summaryLines = lines.slice(lastBoundary).filter(l => l.trim().length > 0);
                    if (summaryLines.length > 0) {
                        pages.push(summaryLines.join('\n'));
                    }
                }
            }
        }
    }
    
    return pages.length > 0 ? pages : [output];
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
    const statusEl = document.getElementById('status');
    const outputEl = document.getElementById('output');
    const paginationEl = document.getElementById('pagination');
    
    runButton.disabled = true;
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
        
        statusEl.className = 'status loading';
        statusEl.innerHTML = '<span class="spinner"></span>Running tests...';
        statusEl.classList.remove('hidden');
        
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
                statusEl.className = 'status success';
                statusEl.textContent = '✓ All tests passed!';
            } else {
                statusEl.className = 'status error';
                statusEl.textContent = '✗ Some tests failed';
            }
            
        } catch (error) {
            clearInterval(updateInterval);
            const errorMsg = error.toString();
            const captured = getPythonOutput();
            outputEl.textContent = (captured || '') + '\n' + errorMsg;
            
            statusEl.className = 'status error';
            statusEl.textContent = `✗ Error running tests: ${errorMsg}`;
        }
        
    } catch (error) {
        statusEl.className = 'status error';
        statusEl.textContent = `Error: ${error.message}`;
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
        await initPyodideRuntime();
        setupPrintCapture();
        
        const statusEl = document.getElementById('status');
        statusEl.className = 'status success';
        statusEl.textContent = '✓ Ready to run tests!';
    } catch (error) {
        console.error('Failed to initialize Pyodide:', error);
    }
});

// Export for use in HTML
window.runTests = runTests;
window.showNextPage = function() { displayPage(currentPageIndex + 1); };
window.showPrevPage = function() { displayPage(currentPageIndex - 1); };
