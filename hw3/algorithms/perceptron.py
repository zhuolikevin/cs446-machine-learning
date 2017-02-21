import numpy

class Perceptron():
    def __init__(self, dimension):
        self.w = [0 for i in range(dimension)]
        self.theta = 0

    def train(self, y, x, eta, gamma=0):
        for i in range(20):
            for j in range(len(y)):
                predictY = numpy.dot(self.w, x[j]) + self.theta
                if y[j] * predictY <= gamma:
                    for k in range(len(self.w)):
                        self.w[k] = self.w[k] + eta * y[j] * x[j][k]
                    self.theta = self.theta + eta * y[j]

    def test(self, y, x):
        correct_num = incorrect_num = 0
        for i in range(len(y)):
            predictY = numpy.sign(numpy.dot(self.w, x[i]) + self.theta)
            if predictY == y[i]:
                correct_num += 1
            else:
                incorrect_num += 1

        accuracy = correct_num * 100.0 / (correct_num + incorrect_num)
        return correct_num, incorrect_num, accuracy

    def train_track_mistakes(self, y, x, eta, gamma=0):
        mistakes = []
        mistake_num = 0
        for j in range(len(y)):
            predictY = numpy.dot(self.w, x[j]) + self.theta

            if y[j] != numpy.sign(predictY):
                mistake_num += 1
            mistakes.append(mistake_num)

            if y[j] * predictY <= gamma:
                for k in range(len(self.w)):
                    self.w[k] = self.w[k] + eta * y[j] * x[j][k]
                self.theta = self.theta + eta * y[j]

        return mistakes, mistake_num
