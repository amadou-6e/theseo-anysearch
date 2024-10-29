#!/bin/bash

# Check if Xvfb is already running on display :99
if ! ps aux | grep -v grep | grep "Xvfb :99" > /dev/null; then
    # Start Xvfb with a 1024x768 screen
    Xvfb :99 -screen 0 1024x768x24 &

    # Wait for Xvfb to start
    sleep 1
fi

# Set the display environment variable
export DISPLAY=:99

# Execute the command passed to the script (if any)
if [ $# -gt 0 ]; then
    exec "$@"
fi