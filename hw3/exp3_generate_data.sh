#!/bin/bash

mkdir data

python data_generator.py 10 100 1000 50000 1 exp3_m100_noise

python data_generator.py 10 500 1000 50000 1 exp3_m500_noise

python data_generator.py 10 1000 1000 50000 1 exp3_m1000_noise

# The below is the clean data set, we only use the `all` set
# `d1` and `d2` will not be used
python data_generator.py 10 100 1000 10000 0 exp3_m100_clean

python data_generator.py 10 500 1000 10000 0 exp3_m500_clean

python data_generator.py 10 1000 1000 10000 0 exp3_m1000_clean
