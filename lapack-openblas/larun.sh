#!/bin/bash

# Define the list of numbers to cycle through
numbers=(5000)

# Use a for loop to iterate over the array and run the command for each number, 10 times
for num in "${numbers[@]}"; do
    echo "Running docker run lapack $num"
    for i in {1..10}; do
        docker run laqr "$num"
    done
done
