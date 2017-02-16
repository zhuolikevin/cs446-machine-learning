package cs446.homework2;

import weka.core.Instances;
import java.io.File;
import java.io.FileReader;
import cs446.weka.classifiers.trees.Id3;
import cs446.homework2.StumpSGD;
import cs446.homework2.SGD;
import cs446.homework2.StatisticalUtil;

public class StumpSGDTester {

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

    System.out.println("========== SGD with Stumps ==========");

    double[] accuracyListCV = new double[5];
    for (int i = 0; i < 5; i++) {
      System.out.println("----- Fold " + (i + 1) + "/5 -----");
      StumpSGD stumpSGD = new StumpSGD();

      stumpSGD.buildStumps(train[i]);
      System.out.println("Building stumps done!");

      Instances curTrainSet = stumpSGD.buildFeaturesFromStumps(train[i]);
      System.out.println("Generating new training set with features from stumps done!");
      Instances curTestSet = stumpSGD.buildFeaturesFromStumps(test[i]);
      System.out.println("Generating new test set with features from stumps done!");

      SGD classifier = new SGD(100);
      classifier.train(curTrainSet, 0.01, 0.00001);
      accuracyListCV[i] = classifier.test(curTestSet);
      System.out.println("Fold Accuracy: " + accuracyListCV[i]);
    }
    System.out.println("----- Summary -----");
    System.out.println("Average Accuracy: " + String.format("%.2f", StatisticalUtil.getAverage(accuracyListCV)) + "%");
    System.out.println("Standard Deviation: " + String.format("%.2f", StatisticalUtil.getStandardDeviation(accuracyListCV)));
  }
}
