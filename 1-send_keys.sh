#!/bin/bash
# Arg 1: maximum number of keys needed

# Reading in variables
source vars.sh

# Creates all the key pairs and sends to the nodes along with host and public files

##########################################################################################
# Read the ip addresses of all the nodes from 'HOSTS_ALL.example'

IFS=$'\n' read -d '' -r -a all_nodes < $HOSTS_FILE

##########################################################################################

# # Locally create all the private and pulbic keys for all the nodes (will be sent to the nodes in the for loop)
# Scripts/setup-ssl.sh "${#all_nodes[@]}"
Scripts/setup-ssl.sh $1

##########################################################################################

# Send private keys
for i in "${!all_nodes[@]}"; do 
    echo ${all_nodes[$i]}

    ##################################################
    # Send (all) private keys to each node (not a safe thing to do...)
    #scp -oStrictHostKeyChecking=no Player-Data/P$i.key $USER@${all_nodes[$i]}:$MP_SPDZ_PATH/Player-Data/
    scp -oStrictHostKeyChecking=no Player-Data/*.key $USER@${all_nodes[$i]}:$MP_SPDZ_PATH/Player-Data/

    ##################################################
    # Send all public keys and auxiliary data to all the nodes
    scp Player-Data/*.pem $USER@${all_nodes[$i]}:$MP_SPDZ_PATH/Player-Data/
    scp Player-Data/*.0 $USER@${all_nodes[$i]}:$MP_SPDZ_PATH/Player-Data/
    
    #################################################
    # Send the folder with the hosts to all nodes
    ssh $USER@${all_nodes[$i]} "cd $MP_SPDZ_PATH && rm -r HOSTS && rm -r Programs/Public-Input && mkdir Programs/Public-Input"
    scp -r HOSTS $USER@${all_nodes[$i]}:$MP_SPDZ_PATH

    #################################################
    # From stackoverflow
    ssh $USER@${all_nodes[$i]} "ulimit -n 4096"

done
