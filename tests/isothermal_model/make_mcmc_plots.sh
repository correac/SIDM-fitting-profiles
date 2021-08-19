#!/bin/bash

# Runs SIDM-fitting scripts using the following variables:

END=1
for ((i=0;i<=END;i++)); do
  input="../../data/L006N188_SigmaConstant10/Individual_sample/Profile_halos_M10.0_DML006N188_SigmaConstant10_$i.txt"
  echo Running for Profile_halos_M10.0_DML006N188_SigmaConstant10_$i.txt
  name="DML006N188_SigmaConstant10_M10.0_$i"
  output="./output/cartesius/"
  python plot_mcmc.py -i=$input -o=$output -n=$name -hs="Indiviual_sample"
done
