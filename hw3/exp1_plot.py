import time
import numpy
import matplotlib.pyplot as plt
from algorithms.perceptron import Perceptron
from algorithms.winnow import Winnow
from algorithms.ada_grad import AdaGrad

start_time = time.time()

t = range(1, 50001)

for exp_flag in ["a", "b"]:
    if exp_flag == "a":
        print "\n>>>>>>>>>> l=10, m=100, n=500 <<<<<<<<<<"
        dimension = 500
        plt.figure(1)
        plt.title("l=10, m=100, n=50")
    else:
        print "\n>>>>>>>>>> l=10, m=100, n=1000 <<<<<<<<<<"
        dimension = 1000
        plt.figure(2)
        plt.title("l=10, m=100, n=1000")

    print "Loading data..."
    y = numpy.load("data/exp1_" + exp_flag + "_all_y.npy")
    x = numpy.load("data/exp1_" + exp_flag + "_all_x.npy")

    # perceptron
    print "\n----- Perceptron -----"

    perceptron = Perceptron(dimension)

    (mistakes_perceptron, mistake_num_perceptron) = perceptron.train_track_mistakes(y, x, 1)

    plt.plot(t, mistakes_perceptron, "r", label="Perceptron")
    print "Accumulated Mistake Numbers: %s" % mistake_num_perceptron
    print "[Time Consumed] %ss" % (time.time() - start_time)

    # perceptron with margin
    print "\n----- Perceptron w/ margin -----"

    perceptron = Perceptron(dimension)
    eta = 0.005 if exp_flag == "a" else 0.03
    gamma = 1
    print "eta = %s" % eta
    print "gamma = %s" % gamma

    (mistakes_perceptron_margin, mistake_num_perceptron_margin) = perceptron.train_track_mistakes(y, x, eta, gamma)

    plt.plot(t, mistakes_perceptron_margin, "b", label="Perceptron w/ margin")
    print "Accumulated Mistake Numbers: %s" % mistake_num_perceptron_margin
    print "[Time Consumed] %ss" % (time.time() - start_time)

    # winnow
    print "\n----- Winnow -----"

    winnow = Winnow(dimension)
    alpha = 1.1
    print "alpha = %s" % alpha

    (mistakes_winnow, mistake_num_winnow) = winnow.train_track_mistakes(y, x, alpha)

    plt.plot(t, mistakes_winnow, "g", label="Winnow")
    print "Accumulated Mistake Numbers: %s" % mistake_num_winnow
    print "[Time Consumed] %ss" % (time.time() - start_time)

    # winnow with margin
    print "\n----- Winnow w/ margin -----"

    winnow = Winnow(dimension)
    alpha = 1.1
    gamma = 0.001 if exp_flag == "a" else 2.0
    print "alpha = %s" % alpha
    print "gamma = %s" % gamma

    (mistakes_winnow_margin, mistake_num_winnow_margin) = winnow.train_track_mistakes(y, x, alpha, gamma)

    plt.plot(t, mistakes_winnow_margin, "y", label="Winnow w/ margin")
    print "Accumulated Mistake Numbers: %s" % mistake_num_winnow_margin
    print "[Time Consumed] %ss" % (time.time() - start_time)

    # adagrad
    print "\n----- AdaGrad -----"

    ada_grad = AdaGrad(dimension)
    eta = 0.25
    print "eta = %s" % eta

    (mistakes_adagrad, mistake_num_adagrad) = ada_grad.train_track_mistakes(y, x, eta)

    plt.plot(t, mistakes_adagrad, "black", label="AdaGrad")
    print "Accumulated Mistake Numbers: %s" % mistake_num_adagrad
    print "[Time Consumed] %ss" % (time.time() - start_time)

    plt.legend(loc=0)
    plt.xlabel('Examples')
    plt.ylabel('Number of Mistakes')
    plt.grid(True)

plt.show()
