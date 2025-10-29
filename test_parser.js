// Quick test of the parseTestOutput function
// Run with: node test_parser.js

const fs = require('fs');

// Simplified version of parseTestOutput from test_runner.js
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
        // Each test includes everything up to (but not including) the next test's separator
        const testLines = lines.slice(startIdx, endIdx);
        
        // Remove trailing blank lines to clean up the output
        while (testLines.length > 0 && testLines[testLines.length - 1].trim() === '') {
            testLines.pop();
        }
        
        if (testLines.length > 0) {
            pages.push(testLines.join('\n'));
        }
    }
    
    return pages.length > 0 ? pages : [output];
}

// Read the test output
try {
    const testOutput = fs.readFileSync('test_output_sample.txt', 'utf8');
    const pages = parseTestOutput(testOutput);
    
    console.log(`Found ${pages.length} test pages\n`);
    console.log('First 5 pages preview:');
    console.log('='.repeat(70));
    for (let i = 0; i < Math.min(5, pages.length); i++) {
        const lines = pages[i].split('\n');
        const testName = lines.find(l => l.includes('TEST:'));
        console.log(`\nPage ${i + 1}: ${testName ? testName.trim() : 'No TEST: found'}`);
        console.log(`  Lines: ${lines.length}`);
        console.log(`  First line: ${lines[0].substring(0, 60)}...`);
    }
    
    console.log(`\n${'='.repeat(70)}`);
    console.log(`Total pages: ${pages.length}`);
    
    // Count how many have TEST: headers
    const testHeaders = pages.filter(p => p.includes('TEST:'));
    console.log(`Pages with TEST: headers: ${testHeaders.length}`);
} catch (error) {
    console.error('Error:', error.message);
}

