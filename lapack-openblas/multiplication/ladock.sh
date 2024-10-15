#!/bin/bash

# Build the Docker container
echo "Building Docker container..."
docker build -t lamul:latest -f Dockerfile.lamul .

# Check if the build was successful
if [ $? -eq 0 ]; then
    echo "Docker build completed successfully."
else
    echo "Docker build failed."
fi
