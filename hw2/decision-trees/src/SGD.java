package cs446.homework2;

import weka.core.Instance;
import weka.core.Instances;

import java.util.Random;
import java.util.Vector;

public class SGD {
  private Vector<Double> w = new Vector<>();

  public SGD(int dimension) {
//    Random randomGenerator = new Random();
    for (int i = 0; i < dimension; i++) {
//      w.add(Math.sin(randomGenerator.nextDouble() * 2 * Math.PI));
      w.add(0.0);
    }
  }

  public Vector<Double> getW() {
    return w;
  }

  public void printW() {
    System.out.println(w.toString());
  }

  public double getPrediction(Vector<Integer> x) throws Exception {
    if (w.size() != x.size()) {
      throw new IndexOutOfBoundsException("w and x should be in the same dimensions");
    }
    double result = 0;
    for (int i = 0; i < w.size(); i++) {
      result += w.elementAt(i) * x.elementAt(i);
    }
    return result;
  }

  public String getPredictionClassified(Vector<Integer> x) throws Exception {
    return getPrediction(x) > 0 ? "+" : "-";
  }

  public void train(Instances data, double learningRate, double threshold) throws Exception {
    boolean stopFlag = false;
    double avgError = Double.POSITIVE_INFINITY;
    while (!stopFlag) {
      int instancesSize = data.numInstances();
      int i;
      for (i = 0; i < instancesSize; i++) {
        Instance curInstance = data.instance(i);
        Vector<Integer> x = resolveInstanceValue(curInstance.toDoubleArray());
        String label = resolveInstanceClass(curInstance.classValue());
        double y = labelToDouble(label);
        for (int j = 0; j < w.size(); j++) {
          double tempWj = w.get(j) - learningRate * x.get(j) * (getPrediction(x) - y);
          w.set(j, tempWj);
        }

        double tempAvgError = calculateAvgError(data);

        if (Math.abs(tempAvgError - avgError) < threshold) {
          stopFlag = true;
          break;
        } else {
          avgError = tempAvgError;
        }
      }
    }
  }

  public double test(Instances data) throws Exception {
    int instancesSize = data.numInstances();
    double correctNum = 0;
    double incorrectNum = 0;
    for (int i = 0; i < instancesSize; i++) {
      Instance curInstance = data.instance(i);
      Vector<Integer> x = resolveInstanceValue(curInstance.toDoubleArray());
      String label = resolveInstanceClass(curInstance.classValue());
      String prediction = getPredictionClassified(x);
      if (prediction == label) {
        correctNum++;
      } else {
        incorrectNum++;
      }
    }
    return correctNum * 100 / (correctNum + incorrectNum);
  }

  public double calculateAvgError(Instances data) throws Exception {
    int instancesSize = data.numInstances();
    double tempSum = 0;
    for (int i = 0; i < instancesSize; i++) {
      Instance curInstance = data.instance(i);
      Vector<Integer> x = resolveInstanceValue(curInstance.toDoubleArray());
      String label = resolveInstanceClass(curInstance.classValue());
      double y = labelToDouble(label);
//      tempSum += Math.pow(labelToDouble(getPredictionClassified(x)) - y, 2);
      tempSum += Math.pow(getPrediction(x) - y, 2);
    }
    return tempSum / instancesSize;
  }

  protected Vector<Integer> resolveInstanceValue(double[] instance) {
    Vector<Integer> result = new Vector<>();

    // Eliminate the last class label
    for (int i = 0; i < instance.length - 1; i++) {
      result.add(instance[i] == 1.0 ? 0 : 1);
    }
    return result;
  }

  protected String resolveInstanceClass(double classValue) {
    return classValue == 0.0 ? "+" : "-";
  }

  protected double labelToDouble(String label) throws Exception {
    if (label == "+") return 1.0;
    else if (label == "-") return -1.0;
    else throw new IllegalArgumentException("Invalid label");
  }
}
