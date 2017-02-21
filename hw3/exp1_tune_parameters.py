import time
import numpy
from tune_parameters import tune_parameters

start_time = time.time()

for exp_flag in ["a", "b"]:
    if exp_flag == "a":
        print "\n>>>>>>>>>> l=10, m=100, n=500 <<<<<<<<<<"
        dimension = 500
    else:
        print "\n>>>>>>>>>> l=10, m=100, n=1000 <<<<<<<<<<"
        dimension = 1000

    print "Loading data..."
    train_y = numpy.load("data/exp1_" + exp_flag + "_d1_y.npy")
    train_x = numpy.load("data/exp1_" + exp_flag + "_d1_x.npy")
    test_y = numpy.load("data/exp1_" + exp_flag + "_d2_y.npy")
    test_x = numpy.load("data/exp1_" + exp_flag + "_d2_x.npy")

    tune_parameters(dimension, start_time, train_y, train_x, test_y, test_x)
