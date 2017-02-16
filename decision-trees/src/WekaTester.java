package cs446.homework2;

import java.io.File;
import java.io.FileReader;

import weka.classifiers.Evaluation;
import weka.core.Instances;
import cs446.weka.classifiers.trees.Id3;

public class WekaTester {
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

    /* ID3 with Unlimited Depth, Depth of 4, Depth of 8 */
    int[] depthList = new int[] { -1, 4, 8 };
    for (int depth : depthList) {
      if (depth == -1) {
        System.out.println("=== ID3 with Unlimited Depth ===");
      } else {
        System.out.println("=== ID3 with Depth of " + depth + " ===");
      }

      Id3 classifier = new Id3();
      classifier.setMaxDepth(depth);

      double[] accuracyListCV = new double[5];
      for (int i = 0; i < 5; i++) {
//        System.out.println("Fold " + (i + 1) + "/5");

        classifier.buildClassifier(train[i]);
        Evaluation evaluation = new Evaluation(test[i]);
        evaluation.evaluateModel(classifier, test[i]);

        double correctInstanceNum = evaluation.correct();
        double incorrectInstanceNum = evaluation.incorrect();

//        accuracyListCV[i] = correctInstanceNum / (correctInstanceNum + incorrectInstanceNum);
        accuracyListCV[i] = evaluation.pctCorrect();
        System.out.println(accuracyListCV[i]);
//        System.out.println("\tCorrectly Classified Instances: " + evaluation.correct());
//        System.out.println("\tIncorrectly Classified Instances: " + evaluation.incorrect());
//        System.out.println("\tAccuracy: " + String.format("%.2f", accuracyListCV[i]) + "%");
      }

      System.out.println("Average Accuracy: " + String.format("%.2f", getAverage(accuracyListCV)) + "%");
      System.out.println("Standard Deviation: " + String.format("%.2f", getStandardDeviation(accuracyListCV)));
    }
  }
}
