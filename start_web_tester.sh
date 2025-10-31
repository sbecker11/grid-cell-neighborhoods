#!/bin/bash
# Start the web tester server on port 8000
# 
# This script will:
# - Check if port 8000 is in use
# - Prompt to kill any processes using port 8000
# - Start the server on port 8000 (or next available port if declined)
# - Display the URL to open in your browser
#
# Usage: ./start_web_tester.sh

cd "$(dirname "$0")"
python3 web_tester/server.py

