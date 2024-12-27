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

# clang
#build_container lapack_qr" "Dockerfile.laqr"
#build_container lapack_svd" "Dockerfile.lasvd"
#build_container lapack_chol" "Dockerfile.lachol"
build_container lapack_mul" "Dockerfile.lamul"
#build_container lapack_qr" "Dockerfile.laqr"
#build_container lapack_gaus_no_omp" "Dockerfile.lagaus"
#build_container lapack_svd_wo_omp" "Dockerfile.lasvd_withoutOMP"
#build_container lapack_qr_wo_omp" "Dockerfile.laqr_withoutOMP"

cd ../
echo "All containers built successfully!"

