package cs446.homework2;

import weka.core.Instances;

import java.io.File;
import java.io.FileReader;

import cs446.homework2.SGD;

public class SGDTester {
  public static double getAverage(double[] array) {
    double sum = 0;
    for (int i = 0; i < array.length; i++) {
      sum += array[i];
    }
    return sum / array.length;
  }

  public static double getStandardDeviation(double[] array) {
    double sum = 0;
    double average = getAverage(array);
    for (int i = 0; i < array.length; i++) {
      sum += (array[i] - average) * (array[i] - average);
    }
    return Math.sqrt(sum / array.length);
  }

  public static void main(String[] args) throws Exception {
    if (args.length != 10) {
      System.err.println("Usage: WekaTester arff-file5 arff-file4 arff-file3 arff-file2 arff-file1" +
              "arff-file1234 arff-file1235 arff-file1245 arff-file1345 arff-file2345");
      System.exit(-1);
    }

    /* Load test and train sets */
    Instances[] test = new Instances[5];
    Instances[] train = new Instances[5];
    for (int i = 0; i < 5; i++) {
      test[i] = new Instances(new FileReader(new File(args[i])));
      test[i].setClassIndex(test[i].numAttributes() - 1);
    }
    for (int i = 0; i < 5; i++) {
      train[i] = new Instances(new FileReader(new File(args[i + 5])));
      train[i].setClassIndex(train[i].numAttributes() - 1);
    }

    double[] learningRates = new double[] { 0.00001, 0.0001, 0.001, 0.01, 0.1 };
    double[] thresholds = new double[] { 0.0000001, 0.0000005, 0.000001, 0.000005, 0.00001 };
    double maxArg = Double.NEGATIVE_INFINITY;
    double standardDeviationCorres = 0;
    double optimalLearningRate = learningRates[0];
    double optimalThreshold = thresholds[0];

    for (int i = 0; i < learningRates.length; i++) {
      for (int j = 0; j < thresholds.length; j++) {
        double[] accuracyListCV = new double[5];
        for (int k = 0; k < 5; k++) {

          SGD classifier = new SGD(260);

          classifier.train(train[k], learningRates[i], thresholds[j]);
          accuracyListCV[k] = classifier.test(test[k]);

//          System.out.println(accuracyListCV[k]);
        }
        double avgAccuracy = getAverage(accuracyListCV);
        if (avgAccuracy > maxArg) {
          maxArg = avgAccuracy;
          standardDeviationCorres = getStandardDeviation(accuracyListCV);
          optimalLearningRate = learningRates[i];
          optimalThreshold = thresholds[j];
        }
      }
    }
    System.out.println("Optimal Learning Rate: " + optimalLearningRate);
    System.out.println("Optimal Threshold: " + optimalThreshold);
    System.out.println("Average Accuracy: " + String.format("%.2f", maxArg) + "%");
    System.out.println("Standard Deviation: " + String.format("%.2f", standardDeviationCorres));

//    double[] accuracyListCV = new double[5];
//    for (int k = 0; k < 5; k++) {
//      SGD classifier = new SGD(260);
//
//      classifier.train(train[k], 0.00001, 0.00000001);
//      accuracyListCV[k] = classifier.test(test[k]);
//
//      System.out.println(accuracyListCV[k]);
//    }
//    System.out.println("Average Accuracy: " + String.format("%.2f", getAverage(accuracyListCV)) + "%");
//    System.out.println("Standard Deviation: " + String.format("%.2f", getStandardDeviation(accuracyListCV)));
  }
}
