#!/bin/bash

# Directory containing test files
test_dir="test_files"

# Loop through each test file in the directory
for test_file in "$test_dir"/test_*; do
    # Extract the base name of the test file
    base_name=$(basename "$test_file")

    # Determine the first input value based on the test file name
    case "$base_name" in
        test_01*)
            input_value=35
            ;;
        test_02*)
            input_value=30000
            ;;
        test_03*|test_04*|test_05*|test_06*)
            input_value=20
            ;;
        test_07*)
            input_value=1000000
            ;;
        test_08*)
            input_value=100000000
            ;;
        *)
            echo "================================================"
            continue
            ;;
    esac
    echo "Running test $test_file with input value $input_value ..."
    # Run the executable with the determined input values
    mpirun -np 2 --bind-to none ./energy_storms_seq "$input_value" "$test_file"
    echo "================================================"
done    

echo "Running test test_files/test_09_a16-17_p3_w1 with input value 16 ..."
mpirun -np 2 --bind-to none ./energy_storms_seq "16" "test_files/test_09_a16-17_p3_w1"
echo "================================================"
echo "Running test test_files/test_09_a16-17_p3_w1 with input value 17 ..."
mpirun -np 2 --bind-to none ./energy_storms_seq "17" "test_files/test_09_a16-17_p3_w1"