#!/bin/bash

# Runs SIDM-fitting scripts using the following variables:

# To extract SIDM halo profiles:
input="/Users/camila/SimulationData/cartesius/L006N188/SigmaConstant00/DMONLY"
output="./data/L006N188_SigmaConstant00/"
snapshot="36"
name="DML006N188_SigmaConstant00"
python extract_profile_cosmo_box.py -i=$input -s=$snapshot -o=$output -n=$name

input="/Users/camila/SimulationData/cartesius/L006N188/SigmaConstant01/DMONLY"
output="./data/L006N188_SigmaConstant01/"
snapshot="36"
name="DML006N188_SigmaConstant01"
python extract_profile_cosmo_box.py -i=$input -s=$snapshot -o=$output -n=$name

#output="./output/"
#sigma=10
#w=30
#n=0
#
#END=4
#for ((i=0;i<=END;i++)); do
#  input="./data/L006N188_SigmaVel100/Profile_halos_M10.0_DML006N188_SigmaVel100_$i.txt"
#  echo Running for Profile_halos_M10.0_DML006N188_SigmaVel100_$i.txt
#  name="DML006N188_SigmaVel100_M10_$i"
#  python main.py -i=$input -o=$output -n=$name -v=$n -d=$sigma -w=$w
#done

#input="./data/L006N188_SigmaVel100/Profile_halos_M9.0_DML006N188_SigmaVel100.txt"
#name="DML006N188_SigmaVel100_M9"
#output="./output/"
#sigma=100
#w=30
#n=0
#python main.py -i=$input -o=$output -n=$name -v=$n -d=$sigma -w=$w
#
#input="./data/L006N188_SigmaVel100/Profile_subhalos_M9.0_DML006N188_SigmaVel100.txt"
#name="DML006N188_SigmaVel100_subM9"
#output="./output/"
#sigma=100
#w=30
#n=0
#python main.py -i=$input -o=$output -n=$name -v=$n -d=$sigma -w=$w

# To run sidm-fitting routine:
# Input / output folders:
#input="./data/L025N376_SigmaVel45/Profile_halos_M10_L025N376_SigmaVel45.txt"
#name="L025N376_SigmaVel45_M10"
#input="./data/FermionDSph_Carina.txt"
#name="FermionDSph_Carina"
#output="./output/"
#sigma=50
#w=30
#n=0
#
#python main.py -i=$input -o=$output -n=$name -v=$n -d=$sigma -w=$w
#mpiexec -np 4 python main.py -i=$input -o=$output -n=$name -v=$n -d=$sigma -w=$w


