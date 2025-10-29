// Test Runner for Grid Cell Neighborhoods
// Uses Pyodide to run Python tests in the browser

let pyodide = null;
let loading = false;

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
            indexURL: "https://cdn.jsdelivr.net/pyodide/v0.24.1/full/"
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
        pyodide.runPython(`
import sys, importlib
if '' not in sys.path:
    sys.path.append('')
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

// Run all tests
async function runTests() {
    if (loading) return;
    
    loading = true;
    const runButton = document.getElementById('runTests');
    const statusEl = document.getElementById('status');
    const outputEl = document.getElementById('output');
    
    runButton.disabled = true;
    outputEl.textContent = '';
    
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
                outputEl.scrollTop = outputEl.scrollHeight;
            }
        }, 100);
        
        // Run tests
        try {
            pyodide.runPython(`
                # Run all tests
                success = run_all_tests()
            `);
            
            // Get final output
            const finalOutput = getPythonOutput();
            
            // Check if tests passed
            const testResult = pyodide.runPython('success');
            
            clearInterval(updateInterval);
            
            outputEl.textContent = finalOutput;
            outputEl.scrollTop = outputEl.scrollHeight;
            
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
