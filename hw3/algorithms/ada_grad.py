import numpy
import math

class AdaGrad():
    def __init__(self, dimension):
        self.w = [0 for i in range(dimension)]
        self.theta = 0

    def train(self, y, x, eta):
        # Sum of gradients' squares
        G_w = [0 for i in range(len(self.w))]
        G_theta = 0
        for i in range(20):
            for j in range(len(y)):
                # Update G
                for k in range(len(G_w)):
                    G_w[k] += math.pow(- y[j] * x[j][k], 2)
                G_theta += math.pow(- y[j], 2)

                predictY = numpy.dot(self.w, x[j]) + self.theta
                if y[j] * predictY <= 1:
                    for k in range(len(self.w)):
                        if G_w[k] == 0:
                            continue
                        else:
                            self.w[k] = self.w[k] + eta * y[j] * x[j][k] / math.sqrt(G_w[k])
                    if G_theta != 0:
                        self.theta = self.theta + eta * y[j] / math.sqrt(G_theta)

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

    def train_track_mistakes(self, y, x, eta):
        mistakes = []
        mistake_num = 0

        G_w = [0 for i in range(len(self.w))]
        G_theta = 0

        for j in range(len(y)):
            for k in range(len(G_w)):
                G_w[k] += math.pow(- y[j] * x[j][k], 2)
            G_theta += math.pow(- y[j], 2)

            predictY = numpy.dot(self.w, x[j]) + self.theta

            if y[j] != numpy.sign(predictY):
                mistake_num += 1
            mistakes.append(mistake_num)

            if y[j] * predictY <= 1:
                for k in range(len(self.w)):
                    if G_w[k] == 0:
                        continue
                    else:
                        self.w[k] = self.w[k] + eta * y[j] * x[j][k] / math.sqrt(G_w[k])
                if G_theta != 0:
                    self.theta = self.theta + eta * y[j] / math.sqrt(G_theta)

        return mistakes, mistake_num
