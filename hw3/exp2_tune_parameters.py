import time
import numpy
from tune_parameters import tune_parameters

start_time = time.time()

for dimension in [40, 80, 120, 160, 200]:
    print "\n>>>>>>>>>> l=10, m=20, n=%s <<<<<<<<<<" % dimension

    print "Loading data..."
    train_y = numpy.load("data/exp2_n" + str(dimension) + "_d1_y.npy")
    train_x = numpy.load("data/exp2_n" + str(dimension) + "_d1_x.npy")
    test_y = numpy.load("data/exp2_n" + str(dimension) + "_d2_y.npy")
    test_x = numpy.load("data/exp2_n" + str(dimension) + "_d2_x.npy")

    tune_parameters(dimension, start_time, train_y, train_x, test_y, test_x)
