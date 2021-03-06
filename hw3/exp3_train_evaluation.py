import time
import numpy
from algorithms.perceptron import Perceptron
from algorithms.winnow import Winnow
from algorithms.ada_grad import AdaGrad

start_time = time.time()

DIMENSION = 1000

for relevant_m in [100, 500, 1000]:
    print "\n>>>>>>>>>> l=10, m=%s, n=1000 <<<<<<<<<<" % relevant_m

    print "Loading data..."
    train_y = numpy.load("data/exp3_m" + str(relevant_m) + "_noise_all_y.npy")
    train_x = numpy.load("data/exp3_m" + str(relevant_m) + "_noise_all_x.npy")
    test_y = numpy.load("data/exp3_m" + str(relevant_m) + "_clean_all_y.npy")
    test_x = numpy.load("data/exp3_m" + str(relevant_m) + "_clean_all_x.npy")

    # perceptron
    print "\n----- Perceptron -----"

    perceptron = Perceptron(DIMENSION)
    eta = 1
    print "eta = %s" % eta

    perceptron.train(train_y, train_x, eta)
    (correct_num, incorrect_num, accuracy) = perceptron.test(test_y, test_x)

    print "Correct Predictions: %s" % correct_num
    print "Mistakes: %s" % incorrect_num
    print "Accuracy: %s" % round(accuracy, 2) + "%"
    print "[Time Consumed] %ss" % (time.time() - start_time)

    # perceptron with margin
    print "\n----- Perceptron w/ margin -----"

    perceptron = Perceptron(DIMENSION)
    eta = 0.03 if relevant_m in [100, 1000] else 1.5
    gamma = 1
    print "eta = %s" % eta
    print "gamma = %s" % gamma

    perceptron.train(train_y, train_x, eta, gamma)
    (correct_num, incorrect_num, accuracy) = perceptron.test(test_y, test_x)

    print "Correct Predictions: %s" % correct_num
    print "Mistakes: %s" % incorrect_num
    print "Accuracy: %s" % round(accuracy, 2) + "%"
    print "[Time Consumed] %ss" % (time.time() - start_time)

    # winnow
    print "\n----- Winnow -----"

    winnow = Winnow(DIMENSION)
    alpha = 1.1
    print "alpha = %s" % alpha

    winnow.train(train_y, train_x, alpha)
    (correct_num, incorrect_num, accuracy) = winnow.test(test_y, test_x)

    print "Correct Predictions: %s" % correct_num
    print "Mistakes: %s" % incorrect_num
    print "Accuracy: %s" % round(accuracy, 2) + "%"
    print "[Time Consumed] %ss" % (time.time() - start_time)

    # winnow with margin
    print "\n----- Winnow w/ margin -----"

    winnow = Winnow(DIMENSION)
    alpha = 1.1
    gamma = {100: 0.006, 500: 0.001, 1000:2.0}[relevant_m]
    print "alpha = %s" % alpha
    print "gamma = %s" % gamma

    winnow.train(train_y, train_x, alpha, gamma)
    (correct_num, incorrect_num, accuracy) = winnow.test(test_y, test_x)

    print "Correct Predictions: %s" % correct_num
    print "Mistakes: %s" % incorrect_num
    print "Accuracy: %s" % round(accuracy, 2) + "%"
    print "[Time Consumed] %ss" % (time.time() - start_time)

    # adagrad
    print "\n----- AdaGrad -----"

    ada_grad = AdaGrad(DIMENSION)
    eta = 0.25 if relevant_m == 100 else 1.5
    print "eta = %s" % eta

    ada_grad.train(train_y, train_x, eta)
    (correct_num, incorrect_num, accuracy) = ada_grad.test(test_y, test_x)

    print "Correct Predictions: %s" % correct_num
    print "Mistakes: %s" % incorrect_num
    print "Accuracy: %s" % round(accuracy, 2) + "%"
    print "[Time Consumed] %ss" % (time.time() - start_time)
