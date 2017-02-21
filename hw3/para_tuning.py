import time
import numpy
from algorithms.perceptron import Perceptron
from algorithms.winnow import Winnow
from algorithms.ada_grad import AdaGrad

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

    # # perceptron
    # print "\n----- Perceptron (eta=1) -----"
    #
    # perceptron = Perceptron(dimension)
    # perceptron.train(train_y, train_x, 1)
    # (correct_num, incorrect_num, accuracy) = perceptron.test(test_y, test_x)
    # print "Accuracy: %s" % round(accuracy, 2) + "%"
    #
    # print "[Time Consumed] %s s" % (time.time() - start_time)
    #
    # # perceptron with margin
    # print "\n----- Perceptron w/ margin -----"
    #
    # learning_rates = [1.5, 0.25, 0.03, 0.005, 0.001]
    # max_accuracy = 0
    # for eta in learning_rates:
    #     print "eta=%s..." % eta
    #     perceptron = Perceptron(dimension)
    #     perceptron.train(train_y, train_x, eta, 1)
    #     (correct_num, incorrect_num, accuracy) = perceptron.test(test_y, test_x)
    #     print "Accuracy: %s" % round(accuracy, 2) + "%"
    #     if accuracy > max_accuracy:
    #         max_accuracy = accuracy
    #         optimal_eta = eta
    # print "Optimal learning rate: %s" % optimal_eta
    # print "Accuracy: %s" % round(max_accuracy, 2) + "%"
    #
    # print "[Time Consumed] %ss" % (time.time() - start_time)
    #
    # # winnow
    # print "\n----- Winnow -----"
    #
    # alphas = [1.1, 1.01, 1.005, 1.0005, 1.0001]
    # max_accuracy = 0
    # for alpha in alphas:
    #     print "alpha=%s..." % alpha
    #     winnow = Winnow(dimension)
    #     winnow.train(train_y, train_x, alpha)
    #     (correct_num, incorrect_num, accuracy) = winnow.test(test_y, test_x)
    #     print "Accuracy: %s" % round(accuracy, 2) + "%"
    #     if accuracy > max_accuracy:
    #         max_accuracy = accuracy
    #         optimal_alpha = alpha
    # print "Optimal promotion/demotion parameter: %s" % optimal_alpha
    # print "Accuracy: %s" % round(max_accuracy, 2) + "%"
    #
    # print "[Time Consumed] %ss" % (time.time() - start_time)
    #
    # # winnow with margin
    # print "\n----- Winnow w/ margin -----"
    #
    # alphas = [1.1, 1.01, 1.005, 1.0005, 1.0001]
    # gammars = [2.0, 0.3, 0.04, 0.006, 0.001]
    # max_accuracy = 0
    # for alpha in alphas:
    #     for gammar in gammars:
    #         print "alpha=%s, gammar=%s..." % (alpha, gammar)
    #         winnow = Winnow(dimension)
    #         winnow.train(train_y, train_x, alpha, gammar)
    #         (correct_num, incorrect_num, accuracy) = winnow.test(test_y, test_x)
    #         print "Accuracy: %s" % round(accuracy, 2) + "%"
    #         if accuracy > max_accuracy:
    #             max_accuracy = accuracy
    #             optimal_alpha = alpha
    #             optimal_gammar = gammar
    # print "Optimal promotion/demotion parameter: %s" % optimal_alpha
    # print "Optimal gammar: %s" % optimal_gammar
    # print "Accuracy: %s" % round(max_accuracy, 2) + "%"
    #
    # print "[Time Consumed] %ss" % (time.time() - start_time)

    # adagrad
    print "\n----- AdaGrad -----"
    learning_rates = [1.5, 0.25, 0.03, 0.005, 0.001]
    max_accuracy = 0
    for eta in learning_rates:
        print "eta=%s..." % eta
        ada_grad = AdaGrad(dimension)
        ada_grad.train(train_y, train_x, eta)
        (correct_num, incorrect_num, accuracy) = ada_grad.test(test_y, test_x)
        print "Accuracy: %s" % round(accuracy, 2) + "%"
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            optimal_eta = eta
    print "Optimal learning rate: %s" % optimal_eta
    print "Accuracy: %s" % round(max_accuracy, 2) + "%"

    print "[Time Consumed] %ss" % (time.time() - start_time)
