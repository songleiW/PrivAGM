#!/bin/bash
# Sends a .mpc file to all nodes and compiles it
# First argument is the name of the program to send to the nodes and compile

# Reading in variables
source vars.sh

IFS=$'\n' read -d '' -r -a all_nodes < $HOSTS_FILE

for i in "${!all_nodes[@]}"; do 
    echo ${all_nodes[$i]}
    #################################################
    # Send the .mpc file (arg1) to all the nodes
    scp -oStrictHostKeyChecking=no Programs/Source/$1.mpc $USER@${all_nodes[$i]}:$MP_SPDZ_PATH/Programs/Source/

    #################################################
    # Send the public input to all the nodes
    scp -r Programs/Public-Input $USER@${all_nodes[$i]}:$MP_SPDZ_PATH/Programs/

    #################################################
    # Compile the .mpc file (arg 1) on all nodes
    ssh $USER@${all_nodes[$i]} "cd $MP_SPDZ_PATH && ./compile.py $1" &

done
wait