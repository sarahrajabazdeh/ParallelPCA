#!/bin/bash

# Compile different versions
g++ -fopenmp -o standard standard.cpp -I /usr/include/eigen3
g++ -fopenmp -o cov cov.cpp -I /usr/include/eigen3
g++ -fopenmp -o eigen eigen.cpp -I /usr/include/eigen3
g++ -fopenmp -o pcaparallel pcaparallel.cpp -I /usr/include/eigen3
g++ -o pcasequential pcasequential.cpp -I /usr/include/eigen3

# Get the number of available cores
MAX_CORES=$(nproc)
echo "Maximum available cores: $MAX_CORES"

# Dataset sizes to test
DATASET_SIZES=(100 1000 10000 100000)
# Core counts to test, up to the maximum available cores
CORE_COUNTS=(1 2 4 8 16 24 32)

# Number of runs for averaging
NUM_RUNS=10

# Clear the results file
echo "" > results.txt

for size in "${DATASET_SIZES[@]}"
do
    echo "Testing dataset size: $size" | tee -a results.txt
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
    echo "Average Sequential Time for dataset size $size: $SEQ_AVG seconds" | tee -a results.txt

    # Run parallel standardization version for different core counts
    for cores in "${CORE_COUNTS[@]}"
    do
        if [ $cores -le $MAX_CORES ]; then
            export OMP_NUM_THREADS=$cores
            echo "Testing with $cores cores (standard)" | tee -a results.txt
            PAR_TIMES=()
            for ((i=0; i<$NUM_RUNS; i++))
            do
                PAR_TIME=$(./standard dataset.csv $size 2 | grep -oP '\d+\.\d+')
                PAR_TIMES+=($PAR_TIME)
            done

            PAR_TOTAL=0
            for time in "${PAR_TIMES[@]}"
            do
                PAR_TOTAL=$(echo $PAR_TOTAL + $time | bc)
            done
            PAR_AVG=$(echo "scale=4; $PAR_TOTAL / $NUM_RUNS" | bc)
            echo "Average Parallel Standardization Time for dataset size $size with $cores cores: $PAR_AVG seconds" | tee -a results.txt
        else
            echo "Skipping $cores cores test (standard) because MAX_CORES is $MAX_CORES" | tee -a results.txt
        fi
    done

    # Run parallel covariance version for different core counts
    for cores in "${CORE_COUNTS[@]}"
    do
        if [ $cores -le $MAX_CORES ]; then
            export OMP_NUM_THREADS=$cores
            echo "Testing with $cores cores (cov)" | tee -a results.txt
            PAR_TIMES=()
            for ((i=0; i<$NUM_RUNS; i++))
            do
                PAR_TIME=$(./cov dataset.csv $size 2 | grep -oP '\d+\.\d+')
                PAR_TIMES+=($PAR_TIME)
            done

            PAR_TOTAL=0
            for time in "${PAR_TIMES[@]}"
            do
                PAR_TOTAL=$(echo $PAR_TOTAL + $time | bc)
            done
            PAR_AVG=$(echo "scale=4; $PAR_TOTAL / $NUM_RUNS" | bc)
            echo "Average Parallel Covariance Time for dataset size $size with $cores cores: $PAR_AVG seconds" | tee -a results.txt
        else
            echo "Skipping $cores cores test (cov) because MAX_CORES is $MAX_CORES" | tee -a results.txt
        fi
    done

    # Run parallel eigenvalue decomposition version for different core counts
    for cores in "${CORE_COUNTS[@]}"
    do
        if [ $cores -le $MAX_CORES ]; then
            export OMP_NUM_THREADS=$cores
            echo "Testing with $cores cores (eigen)" | tee -a results.txt
            PAR_TIMES=()
            for ((i=0; i<$NUM_RUNS; i++))
            do
                PAR_TIME=$(./eigen dataset.csv $size 2 | grep -oP '\d+\.\d+')
                PAR_TIMES+=($PAR_TIME)
            done

            PAR_TOTAL=0
            for time in "${PAR_TIMES[@]}"
            do
                PAR_TOTAL=$(echo $PAR_TOTAL + $time | bc)
            done
            PAR_AVG=$(echo "scale=4; $PAR_TOTAL / $NUM_RUNS" | bc)
            echo "Average Parallel Eigen Time for dataset size $size with $cores cores: $PAR_AVG seconds" | tee -a results.txt
        else
            echo "Skipping $cores cores test (eigen) because MAX_CORES is $MAX_CORES" | tee -a results.txt
        fi
    done

    # Run full parallel version for different core counts
    for cores in "${CORE_COUNTS[@]}"
    do
        if [ $cores -le $MAX_CORES ]; then
            export OMP_NUM_THREADS=$cores
            echo "Testing with $cores cores (pcaparallel)" | tee -a results.txt
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
            echo "Average Full Parallel Time for dataset size $size with $cores cores: $PAR_AVG seconds" | tee -a results.txt
        else
            echo "Skipping $cores cores test (pcaparallel) because MAX_CORES is $MAX_CORES" | tee -a results.txt
        fi
    done
done
