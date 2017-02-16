package cs446.homework2;

import java.io.File;
import java.io.FileReader;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import cs446.weka.classifiers.trees.Id3;
import cs446.homework2.StatisticalUtil;

public class WekaTester {

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
        System.out.println("========== ID3 with Unlimited Depth ==========");
      } else {
        System.out.println("========== ID3 with Depth of " + depth + " ==========");
      }

      Id3 classifier = new Id3();
      classifier.setMaxDepth(depth);

      double[] accuracyListCV = new double[5];
      double maxAcc = accuracyListCV[0];
      int maxAccIndex = 0;
      for (int i = 0; i < 5; i++) {
        System.out.println("----- Fold " + (i + 1) + "/5 -----");

        classifier.buildClassifier(train[i]);
        Evaluation evaluation = new Evaluation(test[i]);
        evaluation.evaluateModel(classifier, test[i]);

        accuracyListCV[i] = evaluation.pctCorrect();
        System.out.println("Fold Accuracy: " + accuracyListCV[i]);
        if (accuracyListCV[i] > maxAcc) {
          maxAcc = accuracyListCV[i];
          maxAccIndex = i;
        }

        if (depth == -1 && i == 2 || depth == 4 && i == 1 || depth == 8 && i == 4) {
          System.out.println(classifier);
        }
      }

      System.out.println("----- Summary -----");
      System.out.println("Best Performance Fold: Index " + maxAccIndex + ", Accuracy: " + String.format("%.2f", maxAcc));
      System.out.println("Average Accuracy: " + String.format("%.2f", StatisticalUtil.getAverage(accuracyListCV)) + "%");
      System.out.println("Standard Deviation: " + String.format("%.2f", StatisticalUtil.getStandardDeviation(accuracyListCV)));
    }
  }
}
