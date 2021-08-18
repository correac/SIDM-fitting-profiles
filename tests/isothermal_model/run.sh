#!/bin/bash

# Runs SIDM-fitting scripts using the following variables:
#input="../../data/L006N188_SigmaConstant00/Profile_halos_M9.5_DML006N188_SigmaConstant00.txt"
#name="DML006N188_SigmaConstant00_M9.5"
#output="./output/"
#python main.py -i=$input -o=$output -n=$name

#END=10
#for ((i=5;i<=END;i++)); do
#  input="../../data/L006N188_SigmaConstant01/Individual_sample/Profile_halos_M9.0_DML006N188_SigmaConstant01_$i.txt"
#  echo Running for Profile_halos_M9.0_DML006N188_SigmaConstant01_$i.txt
#  name="DML006N188_SigmaConstant01_M9_$i"
#  output="./output/Individual_sample/"
#  python main.py -i=$input -o=$output -n=$name -hs="Indiviual_sample"
#done

END=0
for ((i=0;i<=END;i++)); do
  input="../../data/L006N188_SigmaConstant01/Individual_sample/Profile_halos_M11.0_DML006N188_SigmaConstant01_$i.txt"
  echo Running for Profile_halos_M11.0_DML006N188_SigmaConstant01_$i.txt
  name="DML006N188_SigmaConstant01_M11.0_$i"
  output="./output/Individual_sample/"
  python main.py -i=$input -o=$output -n=$name -hs="Indiviual_sample"
done

END=0
for ((i=0;i<=END;i++)); do
  input="../../data/L006N188_SigmaConstant10/Individual_sample/Profile_halos_M11.0_DML006N188_SigmaConstant10_$i.txt"
  echo Running for Profile_halos_M11.0_DML006N188_SigmaConstant10_$i.txt
  name="DML006N188_SigmaConstant10_M11.0_$i"
  output="./output/Individual_sample/"
  python main.py -i=$input -o=$output -n=$name -hs="Indiviual_sample"
done

#END=10
#for ((i=0;i<=END;i++)); do
#  input="../../data/L006N188_SigmaConstant10/Individual_sample/Profile_halos_M10.0_DML006N188_SigmaConstant10_$i.txt"
#  echo Running for Profile_halos_M10.0_DML006N188_SigmaConstant10_$i.txt
#  name="DML006N188_SigmaConstant10_M10.0_$i"
#  output="./output/Individual_sample/"
#  python main.py -i=$input -o=$output -n=$name -hs="Indiviual_sample"
#done