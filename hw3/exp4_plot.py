import time
import numpy
import matplotlib.pyplot as plt
from algorithms.ada_grad import AdaGrad

start_time = time.time()

misclass_error_list_list = []
hinge_loss_list_list = []

for dimension in [40, 80, 120, 160, 200]:
    print "\n>>>>>>>>>> l=10, m=20, n=%s <<<<<<<<<<" % dimension

    print "Loading data...\n"
    y = numpy.load("data/exp4_n" + str(dimension) + "_all_y.npy")
    x = numpy.load("data/exp4_n" + str(dimension) + "_all_x.npy")

    ada_grad = AdaGrad(dimension)
    eta = 1.5

    (misclass_error_list, hinge_loss_list) = ada_grad.train_track_misclass_hinge(y, x, eta)
    misclass_error_list_list.append(misclass_error_list)
    hinge_loss_list_list.append(hinge_loss_list)

    print "[Time Consumed] %ss" % (time.time() - start_time)

t = range(50)

plt.figure(1)
plt.plot(t, misclass_error_list_list[0], "r", label="n=40")
plt.plot(t, misclass_error_list_list[1], "b", label="n=80")
plt.plot(t, misclass_error_list_list[2], "g", label="n=120")
plt.plot(t, misclass_error_list_list[3], "y", label="n=160")
plt.plot(t, misclass_error_list_list[4], "black", label="n=200")
plt.legend(loc=0)
plt.xlabel('Round')
plt.ylabel('Errors')
plt.grid(True)

plt.figure(2)
plt.plot(t, hinge_loss_list_list[0], "r", label="n=40")
plt.plot(t, hinge_loss_list_list[1], "b", label="n=80")
plt.plot(t, hinge_loss_list_list[2], "g", label="n=120")
plt.plot(t, hinge_loss_list_list[3], "y", label="n=160")
plt.plot(t, hinge_loss_list_list[4], "black", label="n=200")
plt.legend(loc=0)
plt.xlabel('Round')
plt.ylabel('Loss')
plt.grid(True)

plt.show()
