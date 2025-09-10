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

build_container "lapack_svd" "Dockerfile.lasvd"
build_container "lapack_chol" "Dockerfile.lachol"
build_container "lapack_lu" "Dockerfile.lapackLU"
build_container "lapack_mul" "Dockerfile.lamul"

build_container "mkl_chol" "Dockerfile.mklcho"
build_container "mkl_svd" "Dockerfile.mklsvd"
build_container "mkl_lu" "Dockerfile.mkllu"
build_container "mkl_mul" "Dockerfile.mklmul"

build_container "num_cho" "Dockerfile.numcho"
build_container "num_lu" "Dockerfile.numlu"
build_container "num_mul" "Dockerfile.nummul"
build_container "num_svd" "Dockerfile.numsvd"


cd ../
echo "All containers built successfully!"

