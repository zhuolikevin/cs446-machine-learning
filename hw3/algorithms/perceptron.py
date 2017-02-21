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

    def train_mistakes_vs_examples(self, y, x, eta, gamma=0):
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

    def train_mistakes_learning_curve(self, y, x, threshold, eta, gamma=0):
        mistake_num = 0
        consecutive_correct_num = 0
        iteration_count = 0

        while consecutive_correct_num < threshold:
            iteration_count += 1
            if iteration_count > 10:
                print ">> Convergence problem! Please reduce R."
                break
            print ">> Train iteration: %s" % iteration_count

            for j in range(len(y)):
                predictY = numpy.dot(self.w, x[j]) + self.theta

                if y[j] != numpy.sign(predictY):
                    mistake_num += 1
                    consecutive_correct_num = 0
                else:
                    consecutive_correct_num += 1
                    if consecutive_correct_num >= threshold: break

                if y[j] * predictY <= gamma:
                    for k in range(len(self.w)):
                        self.w[k] = self.w[k] + eta * y[j] * x[j][k]
                    self.theta = self.theta + eta * y[j]

        return mistake_num
