#!/bin/bash

# Runs SIDM-fitting scripts using the following variables:

# To extract SIDM halo profiles:
#input="/Users/camila/SimulationData/cartesius/L006N188/SigmaVel100/DMONLY"
#output="./data/L006N188_SigmaVel100/"
#snapshot="36"
#name="DML006N188_SigmaVel100"
#python extract_profile_cosmo_box.py -i=$input -s=$snapshot -o=$output -n=$name

#output="./output/"
#sigma=10
#w=30
#n=0
#
#END=50
##for ((i=0;i<=END;i++)); do
##  input="../../data/L006N188_SigmaVel100/Profile_halos_M9.4_DML006N188_SigmaVel100_$i.txt"
##  echo Running for Profile_halos_M9.5_DML006N188_SigmaVel100_$i.txt
##  name="DML006N188_SigmaVel100_M9.5_$i"
##  python main.py -i=$input -o=$output -n=$name -v=$n -d=$sigma -w=$w
##done
#
#for ((i=9;i<=END;i++)); do
#  input="../../data/L006N188_SigmaVel100/Profile_halos_M9.0_DML006N188_SigmaVel100_$i.txt"
#  echo Running for Profile_halos_M9.0_DML006N188_SigmaVel100_$i.txt
#  name="DML006N188_SigmaVel100_M9.0_$i"
#  python main.py -i=$input -o=$output -n=$name -v=$n -d=$sigma -w=$w
#done
#
#END=10
#for ((i=0;i<=END;i++)); do
#  input="../../data/L006N188_SigmaVel100/Profile_halos_M10.0_DML006N188_SigmaVel100_$i.txt"
#  echo Running for Profile_halos_M10.0_DML006N188_SigmaVel100_$i.txt
#  name="DML006N188_SigmaVel100_M10.0_$i"
#  python main.py -i=$input -o=$output -n=$name -v=$n -d=$sigma -w=$w
#done

sigma=100
w=30
n=0

#input="../../data/L006N188_SigmaVel100/Profile_subhalos_M10.0_DML006N188_SigmaVel100.txt"
#name="DML006N188_SigmaVel100_subM9.5"
#output="./output/"
#python main.py -i=$input -o=$output -n=$name -v=$n -d=$sigma -w=$w

input="../../data/L006N188_SigmaConstant01/Profile_subhalos_M9.0_DML006N188_SigmaConstant01.txt"
name="DML006N188_SigmaConstant01_M9.0"
output="./output/"
python main.py -i=$input -o=$output -n=$name -v=$n -d=$sigma -w=$w

#input="../../data/L006N188_SigmaVel100/Profile_subhalos_M9.0_DML006N188_SigmaVel100.txt"
#name="DML006N188_SigmaVel100_subM9.0"
#output="./output/"
#python main.py -i=$input -o=$output -n=$name -v=$n -d=$sigma -w=$w

#input="../../data/L006N188_SigmaVel100/Profile_halos_M9.5_DML006N188_SigmaVel100.txt"
#name="DML006N188_SigmaVel100_M9.5"
#output="./output/"
#python main.py -i=$input -o=$output -n=$name -v=$n -d=$sigma -w=$w

#input="../../data/L006N188_SigmaVel100/Profile_subhalos_M9.5_DML006N188_SigmaVel100.txt"
#name="DML006N188_SigmaVel100_subM9.5"
#output="./output/"
#python main.py -i=$input -o=$output -n=$name -v=$n -d=$sigma -w=$w
#
#input="../../data/L006N188_SigmaVel100/Profile_halos_M10.0_DML006N188_SigmaVel100.txt"
#name="DML006N188_SigmaVel100_M10.0"
#output="./output/"
#python main.py -i=$input -o=$output -n=$name -v=$n -d=$sigma -w=$w
#
#input="../../data/L006N188_SigmaVel100/Profile_subhalos_M10.0_DML006N188_SigmaVel100.txt"
#name="DML006N188_SigmaVel100_subM10.0"
#output="./output/"
#python main.py -i=$input -o=$output -n=$name -v=$n -d=$sigma -w=$w

#input="../../data/L006N188_SigmaVel100/Profile_halos_M10.0_DML006N188_SigmaVel100.txt"
#name="DML006N188_SigmaVel100_M10"
#output="./output/"
#sigma=1
#w=1
#n=0
#python main.py -i=$input -o=$output -n=$name -v=$n -d=$sigma -w=$w

#input="../../data/L006N188_SigmaVel100/Profile_subhalos_M10.0_DML006N188_SigmaVel100.txt"
#name="DML006N188_SigmaVel100_subM10"
#output="./output/"
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
