#!/bin/bash

# Build the Docker container
echo "Building Docker container..."
docker build -t intel/oneapi-basekit:latest -f Dockerfile.mkl .

# Check if the build was successful
if [ $? -eq 0 ]; then
    echo "Docker build completed successfully."
else
    echo "Docker build failed."
fi