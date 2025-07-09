#!/bin/bash
# Runs the shamir protocol for the given protocol using the ip's in HOSTS_$size.example
# First argument: name of the program to run
# Second argument: name of the ip file name
# Third argument: protocol name (shamir/malicious-shamir)

source vars.sh

RUNS=1

FILENAME=$HOST_DIR/$2.example

IFS=$'\n' read -d '' -r -a nodes < $FILENAME

for ((j=0;j<$RUNS;j++));do
    # echo required to parse logs
    echo run=$j

    for i in "${!nodes[@]}"; do 
        ssh $USER@${nodes[$i]} "ulimit -n 4096 && cd $MP_SPDZ_PATH && ./$3-party.x -N "${#nodes[@]}" -ip $FILENAME $i $1" &
    done
    wait

done

