#!/bin/bash

mkdir data

python data_generator.py 10 100 500 50000 0 exp1_a

python data_generator.py 10 100 1000 50000 0 exp1_b
