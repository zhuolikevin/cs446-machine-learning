#!/bin/bash

mkdir data

python data_generator.py 10 20 40 10000 1 exp4_n40
python data_generator.py 10 20 80 10000 1 exp4_n80
python data_generator.py 10 20 120 10000 1 exp4_n120
python data_generator.py 10 20 160 10000 1 exp4_n160
python data_generator.py 10 20 200 10000 1 exp4_n200
