#!/bin/bash

# Runs SIDM-fitting scripts using the following variables:

# To extract SIDM halo profiles:
#input="/Users/camila/SimulationData/cartesius/L025N376/SigmaVel45"
#output="./data/L025N376_SigmaVel45/"
#snapshot="60"
#name="L025N376_SigmaVel45"
#python extract_profile_cosmo_box.py -i=$input -s=$snapshot -o=$output -n=$name


# To run sidm-fitting routine:
# Input / output folders:
input="./data/L025N376_SigmaVel45/Profile_halos_M10_L025N376_SigmaVel45.txt"
output="./output/"

# Production name
name="L025N376_SigmaVel45_M10"

# Initial guess
sigma=10
w=10
n=0.0

python main.py -i=$input -o=$output -n=$name -v=$n -d=$sigma -w=$w
#mpiexec -np 4 python main.py -i=$input -o=$output -n=$name -s=$sigma


