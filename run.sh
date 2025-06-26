#!/bin/bash

# Function to stop servers
stop_servers() {
    if [ -f "http_server.pid" ]; then
        echo "Stopping HTTP server..."
        kill $(cat http_server.pid)
        rm http_server.pid
    fi
}

# Function to kill processes on specific ports
kill_port_processes() {
    echo "Checking for processes on ports 8005 and 8000..."
    
    # Kill processes on port 8005
    if lsof -ti:8005 > /dev/null 2>&1; then
        echo "Killing processes on port 8005..."
        lsof -ti:8005 | xargs kill -9
    fi
    
    # Kill processes on port 8000
    if lsof -ti:8000 > /dev/null 2>&1; then
        echo "Killing processes on port 8000..."
        lsof -ti:8000 | xargs kill -9
    fi
    
    echo "Port cleanup completed."
}

# Handle Ctrl+C
trap 'stop_servers; exit' INT

# Stop any existing servers
stop_servers

# Kill processes on ports 8005 and 8000
kill_port_processes

# Start HTTP server in background and save PID
python3 -m http.server 8005 &
echo $! > http_server.pid

echo "Servers started!"
echo "HTTP server running on http://localhost:8005"
echo "API server running on http://localhost:8000"
echo "Access the interface at: http://localhost:8005/api_client.html"
echo "Press Ctrl+C to stop servers"

# Start API server (this will run in foreground)
python3.10 api_server.py 
