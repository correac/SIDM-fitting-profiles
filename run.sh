#!/bin/bash

# Runs SIDM-fitting scripts using the following variables:

#input="./data/FermionDSph_Carina.txt"
#output="./output/"
#name="FermionDSph_Carina"
#sigma=10

#input="./data/Profile_halo_M12_L025N376_sigma_10.txt"
#output="./output/"
#name="L025N376_M12_sigma_cte10"
#sigma=10

input="./data/L025N256_sigma_vel_15/Profile_halos_M12_L025N256.txt"
output="./output/"
name="L025N256_M12_sigma_15"
sigma=15

python main.py -i=$input -o=$output -n=$name -s=$sigma


