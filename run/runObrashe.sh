#!/bin/bash
# Define the list of numbers to cycle through
numbers=(100 500 1000 5000 10000 20000)
# Use a for loop to iterate over the array and run the command for each number, 10 times
for num in "${numbers[@]}"; do
    echo "Running docker run lapack $num"
    for i in {1..10}; do
        docker run lapack "$num"
    done
done


numbers=(100 500 1000 5000 10000 20000)
# Use a for loop to iterate over the array and run the command for each number, 10 times
for num in "${numbers[@]}"; do
    echo "Running docker run intel/oneapi-basekit $num"
    for i in {1..10}; do
        docker run intel/oneapi-basekit "$num"
    done
done


numbers=(100 500 1000 5000 10000 20000)
# Use a for loop to iterate over the array and run the command for each number, 10 times
for n in "${numbers[@]}"; do
    echo "Running docker run num $n"
    for i in {1..10}; do
        docker run num "$n"
    done
done
