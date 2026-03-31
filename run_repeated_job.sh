#!/bin/bash

for dataset in DBLP5 DBLP3 Reddit bitcoinOTC UCI
do
    for i in $(seq 1 3)  # Three repetitions
    do
        echo "Starting run $i on dataset $dataset..."
        python3 main.py $dataset --incremental_learning True
    done
done

for i in $(seq 1 3)  # Three repetitions
do
    echo "Starting run $i on dataset AS-733..."
    python3 main.py as733 as733.yaml --incremental_learning True
done