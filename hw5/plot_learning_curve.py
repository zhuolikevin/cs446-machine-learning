import NN, data_loader, perceptron
import matplotlib.pyplot as plt

# training_data, test_data = data_loader.load_circle_data()
training_data, test_data = data_loader.load_mnist_data()

# domain = 'circles'
# batch_size = 10
# learning_rate = 0.1
# activation_function = 'relu'
# hidden_layer_width = 10
# data_dim = len(training_data[0][0])

domain = 'mnist'
batch_size = 10
learning_rate = 0.1
activation_function = 'tanh'
hidden_layer_width = 10
data_dim = len(training_data[0][0])

net = NN.create_NN(domain, batch_size, learning_rate, activation_function, hidden_layer_width)
learning_curve_data_net = net.train_with_learning_curve(training_data)
iterations_net = []
accuracy_net = []
for i in range(len(learning_curve_data_net)):
    iterations_net.append(learning_curve_data_net[i][0])
    accuracy_net.append(learning_curve_data_net[i][1])
plt.plot(iterations_net, accuracy_net, 'r', label='NN')

perc = perceptron.Perceptron(data_dim)
learning_curve_data_perc = perc.train_with_learning_curve(training_data)
iterations_perc = []
accuracy_perc = []
for i in range(len(learning_curve_data_perc)):
    iterations_perc.append(learning_curve_data_perc[i][0])
    accuracy_perc.append(learning_curve_data_perc[i][1] * 100)
plt.plot(iterations_perc, accuracy_perc, 'b', label='Perceptron')

plt.legend(loc=0)
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()
