package cs446.homework2;

import weka.core.Instances;
import java.io.File;
import java.io.FileReader;
import cs446.homework2.SGD;
import cs446.homework2.StatisticalUtil;

public class SGDTester {

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

//    double[] learningRates = new double[] { 0.00001, 0.0001, 0.001, 0.01, 0.1 };
    double[] learningRates = new double[334];
    for (int i = 0; i < 334; i++) {
      learningRates[i] = 0.0001 + i * 0.0003;
    }
    double[] thresholds = new double[] { 0.000001, 0.00001, 0.0001 };
    double maxArg = Double.NEGATIVE_INFINITY;
    double standardDeviationCorres = 0;
    double optimalLearningRate = learningRates[0];
    double optimalThreshold = thresholds[0];

    System.out.println("========== SGD ==========");

    for (int i = 0; i < learningRates.length; i++) {
      for (int j = 0; j < thresholds.length; j++) {
        System.out.println(">>>>> Progress: " + (i * thresholds.length + j + 1) + "/" + learningRates.length * thresholds.length + "<<<<<");
        System.out.println("Learning Rate: " + learningRates[i]);
        System.out.println("Threshold: " + thresholds[j]);
        double[] accuracyListCV = new double[5];
        for (int k = 0; k < 5; k++) {
          System.out.println("----- Fold " + (k + 1) + "/5 -----");
          SGD classifier = new SGD(260);

          classifier.train(train[k], learningRates[i], thresholds[j]);
          accuracyListCV[k] = classifier.test(test[k]);

          System.out.println("Fold Accuracy: " + accuracyListCV[k]);
        }
        double avgAccuracy = StatisticalUtil.getAverage(accuracyListCV);
        if (avgAccuracy > maxArg) {
          maxArg = avgAccuracy;
          standardDeviationCorres = StatisticalUtil.getStandardDeviation(accuracyListCV);
          optimalLearningRate = learningRates[i];
          optimalThreshold = thresholds[j];
        }
      }
    }
    System.out.println("----- Summary -----");
    System.out.println("Optimal Learning Rate: " + optimalLearningRate);
    System.out.println("Optimal Threshold: " + optimalThreshold);
    System.out.println("Average Accuracy: " + String.format("%.2f", maxArg) + "%");
    System.out.println("Standard Deviation: " + String.format("%.2f", standardDeviationCorres));

  }
}
