import time
import numpy
import matplotlib.pyplot as plt
from algorithms.perceptron import Perceptron
from algorithms.winnow import Winnow
from algorithms.ada_grad import AdaGrad

CONVERGENCE_CRITERION_R = 1000

start_time = time.time()

mistakes_perceptron = []
mistakes_perceptron_margin = []
mistakes_winnow = []
mistakes_winnow_margin = []
mistakes_adagrad = []

for dimension in [40, 80, 120, 160, 200]:
    print "\n>>>>>>>>>> l=10, m=20, n=%s <<<<<<<<<<" % dimension

    print "Loading data..."
    y = numpy.load("data/exp2_n" + str(dimension) + "_all_y.npy")
    x = numpy.load("data/exp2_n" + str(dimension) + "_all_x.npy")

    # perceptron
    print "\n----- Perceptron -----"

    perceptron = Perceptron(dimension)

    mistakes_perceptron.append(perceptron.train_mistakes_learning_curve(y, x, CONVERGENCE_CRITERION_R, 1))

    print "Mistake Numbers: %s" % mistakes_perceptron[-1]
    print "[Time Consumed] %ss" % (time.time() - start_time)

    # perceptron with margin
    print "\n----- Perceptron w/ margin -----"

    perceptron = Perceptron(dimension)
    eta = 0.25 if dimension in [40, 80, 120] else 0.03
    gamma = 1
    print "eta = %s" % eta
    print "gamma = %s" % gamma

    mistakes_perceptron_margin.append(perceptron.train_mistakes_learning_curve(y, x, CONVERGENCE_CRITERION_R, eta, gamma))

    print "Mistake Numbers: %s" % mistakes_perceptron_margin[-1]
    print "[Time Consumed] %ss" % (time.time() - start_time)

    # winnow
    print "\n----- Winnow -----"

    winnow = Winnow(dimension)
    alpha = 1.1
    print "alpha = %s" % alpha

    mistakes_winnow.append(winnow.train_mistakes_learning_curve(y, x, CONVERGENCE_CRITERION_R, alpha))

    print "Mistake Numbers: %s" % mistakes_winnow[-1]
    print "[Time Consumed] %ss" % (time.time() - start_time)

    # winnow with margin
    print "\n----- Winnow w/ margin -----"

    winnow = Winnow(dimension)
    alpha = 1.1
    gamma = 2.0
    print "alpha = %s" % alpha
    print "gamma = %s" % gamma

    mistakes_winnow_margin.append(winnow.train_mistakes_learning_curve(y, x, CONVERGENCE_CRITERION_R, alpha, gamma))

    print "Mistake Numbers: %s" % mistakes_winnow_margin[-1]
    print "[Time Consumed] %ss" % (time.time() - start_time)

    # adagrad
    print "\n----- AdaGrad -----"

    ada_grad = AdaGrad(dimension)
    eta = 1.5
    print "eta = %s" % eta

    mistakes_adagrad.append(ada_grad.train_mistakes_learning_curve(y, x, CONVERGENCE_CRITERION_R, eta))

    print "Mistake Numbers: %s" % mistakes_adagrad[-1]
    print "[Time Consumed] %ss" % (time.time() - start_time)

t = [40, 80, 120, 160, 200]
plt.plot(t, mistakes_perceptron, "r", label="Perceptron")
plt.plot(t, mistakes_perceptron_margin, "b", label="Perceptron w/ margin")
plt.plot(t, mistakes_winnow, "g", label="Winnow")
plt.plot(t, mistakes_winnow_margin, "y", label="Winnow w/ margin")
plt.plot(t, mistakes_adagrad, "black", label="AdaGrad")

plt.legend(loc=0)
plt.xlabel('Dimensions n')
plt.ylabel('Number of Mistakes before Stops')
plt.grid(True)
plt.show()
