import time
import numpy
from tune_parameters import tune_parameters

start_time = time.time()

DIMENSION = 1000

for relevant_m in [100, 500, 1000]:
    print "\n>>>>>>>>>> l=10, m=%s, n=1000 <<<<<<<<<<" % relevant_m

    print "Loading data..."
    train_y = numpy.load("data/exp3_m" + str(relevant_m) + "_noise_d1_y.npy")
    train_x = numpy.load("data/exp3_m" + str(relevant_m) + "_noise_d1_x.npy")
    test_y = numpy.load("data/exp3_m" + str(relevant_m) + "_noise_d2_y.npy")
    test_x = numpy.load("data/exp3_m" + str(relevant_m) + "_noise_d2_x.npy")

    tune_parameters(DIMENSION, start_time, train_y, train_x, test_y, test_x)
