import time
import numpy
from algorithms.perceptron import Perceptron

# exp1a
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

    # perceptron
    print "\n----- Perceptron (eta=1) -----"
    start_time = time.time()

    perceptron = Perceptron(dimension)
    perceptron.train(train_y, train_x, 1)
    (correct_num, incorrect_num, accuracy) = perceptron.test(test_y, test_x)
    print "Accuracy: %s" % round(accuracy, 2) + "%"

    print "[Time Consumption] %s s" % (time.time() - start_time)

    # perceptron with margin
    print "\n----- Perceptron w/ margin -----"
    start_time = time.time()

    learning_rates = [1.5, 0.25, 0.03, 0.005, 0.001]
    max_accuracy = 0
    for eta in learning_rates:
        print "eta=%s..." % eta
        perceptron = Perceptron(dimension, 1)
        perceptron.train(train_y, train_x, eta)
        (correct_num, incorrect_num, accuracy) = perceptron.test(test_y, test_x)
        print "Accuracy: %s" % round(accuracy, 2) + "%"
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            optimal_eta = eta
    print "Optimal learning rate: %s" % optimal_eta
    print "Accuracy: %s" % round(max_accuracy, 2) + "%"

    print "[Time Consumption] %ss" % (time.time() - start_time)
