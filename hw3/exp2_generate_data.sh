#!/bin/bash

mkdir data

python data_generator.py 10 20 40 50000 0 exp2_n40

python data_generator.py 10 20 80 50000 0 exp2_n80

python data_generator.py 10 20 120 50000 0 exp2_n120

python data_generator.py 10 20 160 50000 0 exp2_n160

python data_generator.py 10 20 200 50000 0 exp2_n200
