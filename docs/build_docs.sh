#!/bin/bash

# Build MILSensors documentation
# This script builds the Sphinx documentation and opens it in a browser

echo "Building MILSensors documentation..."

# Install documentation dependencies if needed
if [ ! -d "_build" ]; then
    echo "Installing documentation dependencies..."
    pip install -r requirements.txt
fi

# Build HTML documentation
echo "Building HTML documentation..."
make html

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "Documentation built successfully!"
    echo "HTML files are in _build/html/"
    
    # Try to open in browser (works on macOS and Linux)
    if command -v open &> /dev/null; then
        echo "Opening documentation in browser..."
        open _build/html/index.html
    elif command -v xdg-open &> /dev/null; then
        echo "Opening documentation in browser..."
        xdg-open _build/html/index.html
    else
        echo "Please open _build/html/index.html in your browser to view the documentation."
    fi
else
    echo "Documentation build failed. Please check the error messages above."
    exit 1
fi
