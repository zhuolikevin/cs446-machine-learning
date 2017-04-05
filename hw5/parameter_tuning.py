import NN, data_loader, perceptron
import numpy as np

data_dict = {
    'circles': {
        'training_data': None,
        'test_data': None
    },
    'mnist': {
        'training_data': None,
        'test_data': None
    }
}
(data_dict['circles']['training_data'], data_dict['circles']['test_data']) = data_loader.load_circle_data()
(data_dict['mnist']['training_data'], data_dict['mnist']['test_data']) = data_loader.load_mnist_data()

# Parameters to tune
domains = ['circles', 'mnist']
batch_sizes = [10, 50, 100]
learning_rates = [0.1, 0.01]
activation_functions = ['relu', 'tanh']
hidden_layer_widths = [10, 50]

for domain in domains:
    print '====================', domain, '===================='
    training_data = data_dict[domain]['training_data']
    np.random.shuffle(training_data)
    folder_length = len(training_data) / 5 + 1
    foldered_training_data = [training_data[x:x+folder_length] for x in range(0, len(training_data), folder_length)]

    max_accuracy = 0
    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            for activation_function in activation_functions:
                for hidden_layer_width in hidden_layer_widths:
                    print '---------- size =', batch_size, ', R =', learning_rate, ', func =', activation_function, ', node# =', hidden_layer_width, ' ----------'

                    acc_sum = 0
                    for i in range(5):
                        print '..... Fold', i + 1 , '.....'
                        test_folder = foldered_training_data[i]
                        train_folders = []
                        for j in range(5):
                            if j != i:
                                for item in foldered_training_data[j]:
                                    train_folders.append(item)
                        train_folders = np.array(train_folders)

                        net = NN.create_NN(domain, batch_size, learning_rate, activation_function, hidden_layer_width)
                        net.train(train_folders)

                        acc_sum += net.evaluate(test_folder)
                    print '<Average Accuracy> ', acc_sum / 5.0

                    if acc_sum / 5.0 > max_accuracy:
                        optimal_batch_size = batch_size
                        optimal_learning_rate = learning_rate
                        optimal_activation_function = activation_function
                        optimal_hidden_layer_width = hidden_layer_width
                        max_accuracy = acc_sum / 5.0

    print '[Optimal Batch Size] ', optimal_batch_size
    print '[Optimal Learning Rate] ', optimal_learning_rate
    print '[Optimal Activation Function] ', optimal_activation_function
    print '[Optimal Hidden Layer Width] ', optimal_hidden_layer_width
    print '[Maximum Accuracy] ', max_accuracy
