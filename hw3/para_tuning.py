import time
import numpy
from algorithms.perceptron import Perceptron

# exp1a
print ">>>>>>>>>> l=10, m=100, n=500 <<<<<<<<<<"
print "Loading data..."
train_y = numpy.load("data/exp1_a_d1_y.npy")
train_x = numpy.load("data/exp1_a_d1_x.npy")
test_y = numpy.load("data/exp1_a_d2_y.npy")
test_x = numpy.load("data/exp1_a_d2_x.npy")

# perceptron
print "----- Perceptron (eta=1) -----"
start_time = time.time()

perceptron = Perceptron(500)
perceptron.train(train_y, train_x, 1)
(correct_num, incorrect_num, accuracy) = perceptron.test(test_y, test_x)
print "Accuracy: %s" % round(accuracy, 2) + "%"

print "[Time Consumption] %s s" % (time.time() - start_time)

# perceptron with margin
print "----- Perceptron w/ margin -----"
start_time = time.time()

learning_rates = [1.5, 0.25, 0.03, 0.005, 0.001]
max_accuracy = 0
for eta in learning_rates:
    print "eta=%s..." % eta
    perceptron = Perceptron(500, 1)
    perceptron.train(train_y, train_x, eta)
    (correct_num, incorrect_num, accuracy) = perceptron.test(test_y, test_x)
    print "Accuracy: %s" % round(accuracy, 2) + "%"
    if accuracy > max_accuracy:
        max_accuracy = accuracy
        optimal_eta = eta
print "Optimal learning rate: %s" % optimal_eta
print "Accuracy: %s" % round(max_accuracy, 2) + "%"

print "[Time Consumption] %ss" % (time.time() - start_time)
