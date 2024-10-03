#!/bin/bash

# Build the Docker container
echo "Building Docker container..."
docker build -t eigen:latest -f Dockerfile.Eigen .

# Check if the build was successful
if [ $? -eq 0 ]; then
    echo "Docker build completed successfully."
else
    echo "Docker build failed."
fi

echo "Building Docker container..."
docker build -t lapack:latest -f Dockerfile.lablas .

# Check if the build was successful
if [ $? -eq 0 ]; then
    echo "Docker build completed successfully."
else
    echo "Docker build failed."
fi

echo "Building Docker container..."
docker build -t intel/oneapi-basekit:latest -f Dockerfile.mkl .

# Check if the build was successful
if [ $? -eq 0 ]; then
    echo "Docker build completed successfully."
else
    echo "Docker build failed."
fi

echo "Building Docker container..."
docker build -t num:latest -f Dockerfile.num .

# Check if the build was successful
if [ $? -eq 0 ]; then
    echo "Docker build completed successfully."
else
    echo "Docker build failed."
fi
