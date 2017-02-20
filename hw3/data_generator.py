import sys
import random
import numpy
from gen import gen

DATA_PATH_PREFIX = "data/"
SAMPLE_PERCENTAGE = 0.1

def main(argv):
    if len(argv) != 6:
        print "[ERROR] Run in format - python data_generator.py [l] [m] [n] [instance#] [noise] [filename]"
        sys.exit()
    else:
        l = int(argv[0])
        m = int(argv[1])
        n = int(argv[2])
        instance_num = int(argv[3])
        noise_flag = int(argv[4]) > 0
        filename = argv[5]

    generateData(l, m, n, instance_num, noise_flag, filename);

def generateData(l, m, n, instance_num, noise_flag, filename):
    (y, x) = gen(l, m, n, instance_num, noise_flag);

    sample_num = int(SAMPLE_PERCENTAGE * instance_num)

    instances = zip(x, y)
    sampled_instances = random.sample(instances, sample_num * 2)
    random.shuffle(sampled_instances)

    train_instances = sampled_instances[:sample_num]
    test_instances = sampled_instances[sample_num:]

    (train_x, train_y) = zip(*train_instances)
    (test_x, test_y) = zip(*test_instances)

    numpy.save(DATA_PATH_PREFIX + filename + "_d1_y", list(train_y));
    print "Generated " + DATA_PATH_PREFIX + filename + "_d1_y.npy"

    numpy.save(DATA_PATH_PREFIX + filename + "_d1_x", list(train_x));
    print "Generated " + DATA_PATH_PREFIX + filename + "_d1_x.npy"

    numpy.save(DATA_PATH_PREFIX + filename + "_d2_y", list(test_y));
    print "Generated " + DATA_PATH_PREFIX + filename + "_d2_y.npy"

    numpy.save(DATA_PATH_PREFIX + filename + "_d2_x", list(test_x));
    print "Generated " + DATA_PATH_PREFIX + filename + "_d2_x.npy"

if __name__ == "__main__":
    main(sys.argv[1:])
