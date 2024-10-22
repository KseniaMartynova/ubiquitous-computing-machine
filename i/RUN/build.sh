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
build_container "lapack_qr_clang" "Dockerfile.lablas.QR.clang"
build_container "lapack_lu_clang" "Dockerfile.lablas.LU.clang"
build_container "lapack_ch_clang" "Dockerfile.lablas.Ch.clang"

# g++
build_container "lapack_qr" "Dockerfile.lablas.QR"
build_container "lapack_lu" "Dockerfile.lablas.LU"
build_container "lapack_ch" "Dockerfile.lablas.Ch"

# MKL
echo "Building MKL Docker containers..."

# clang
build_container "mkl_qr_clang" "Dockerfile.mkl.QR.clang"
build_container "mkl_lu_clang" "Dockerfile.mkl.LU.clang"
build_container "mkl_ch_clang" "Dockerfile.mkl.Ch.clang"

# icpx
build_container "mkl_qr" "Dockerfile.mkl.QR"
build_container "mkl_lu" "Dockerfile.mkl.LU"
build_container "mkl_ch" "Dockerfile.mkl.Ch"

# Numpy
build_container "num_ch" "Dockerfile.numCh"
build_container "num_qr" "Dockerfile.numQR"
build_container "num_lu" "Dockerfile.numLU"

cd ../
echo "All containers built successfully!"

