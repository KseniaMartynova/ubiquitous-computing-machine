#!/bin/bash
# Build the Docker container

cd build

# Function to build Docker container with error handling
build_container() {
    local container_name=$1
    local dockerfile=$2

    echo "Building Docker container: $container_name using $dockerfile..."
    if ! docker build -t "$container_name" -f "$dockerfile" .; then
        echo "Error: Building $container_name with $dockerfile failed."
        exit 1
    fi
}

# Lapack-OpenBlas
echo "Building Lapack-OpenBlas Docker containers..."

build_container "lapack_mul" "Dockerfile.lamul"


cd ../
echo "All containers built successfully!"
