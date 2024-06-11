#!/bin/bash

# Compile both versions
g++ -o pcasequential pcasequential.cpp 
g++ -fopenmp -o pcaparallel pcaparallel.cpp

# Dataset sizes to test
DATASET_SIZES=(100 1000 10000 100000)
# Core counts to test
CORE_COUNTS=(1 2 4 8 16)

# Number of runs for averaging
NUM_RUNS=10

for size in "${DATASET_SIZES[@]}"
do
    echo "Testing dataset size: $size"
    # Generate dataset of given size
    python3 data.py $size
    
    # Run sequential version
    SEQ_TIMES=()
    for ((i=0; i<$NUM_RUNS; i++))
    do
        SEQ_TIME=$(./pcasequential dataset.csv $size 2 | grep -oP '\d+\.\d+')
        SEQ_TIMES+=($SEQ_TIME)
    done
    
    SEQ_TOTAL=0
    for time in "${SEQ_TIMES[@]}"
    do
        SEQ_TOTAL=$(echo $SEQ_TOTAL + $time | bc)
    done
    SEQ_AVG=$(echo "scale=4; $SEQ_TOTAL / $NUM_RUNS" | bc)
    echo "Average Sequential Time for dataset size $size: $SEQ_AVG seconds"
    
    # Run parallel version for different core counts
    for cores in "${CORE_COUNTS[@]}"
    do
        export OMP_NUM_THREADS=$cores
        echo "Testing with $cores cores"
        PAR_TIMES=()
        for ((i=0; i<$NUM_RUNS; i++))
        do
            PAR_TIME=$(./pcaparallel dataset.csv $size 2 | grep -oP '\d+\.\d+')
            PAR_TIMES+=($PAR_TIME)
        done
        
        PAR_TOTAL=0
        for time in "${PAR_TIMES[@]}"
        do
            PAR_TOTAL=$(echo $PAR_TOTAL + $time | bc)
        done
        PAR_AVG=$(echo "scale=4; $PAR_TOTAL / $NUM_RUNS" | bc)
        echo "Average Parallel Time for dataset size $size with $cores cores: $PAR_AVG seconds"
    done
done
