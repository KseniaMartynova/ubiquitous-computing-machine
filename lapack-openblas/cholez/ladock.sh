#!/bin/bash

# Build the Docker container
echo "Building Docker container..."
docker build -t lachol:latest -f Dockerfile.lachol .

# Check if the build was successful
if [ $? -eq 0 ]; then
    echo "Docker build completed successfully."
else
    echo "Docker build failed."
fi
