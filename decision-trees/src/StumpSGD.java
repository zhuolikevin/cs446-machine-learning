package cs446.homework2;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import java.util.Random;

import cs446.weka.classifiers.trees.Id3;

public class StumpSGD {

  private Id3[] stumps = new Id3[100];
  public StumpSGD() {}

  static String[] features = new String[100];
  private static FastVector zeroOne;
  private static FastVector labels;

  static {
    for (int i = 0; i < 100; i++) {
      features[i] = "feature" + i;
    }

    zeroOne = new FastVector(2);
    zeroOne.addElement("1");
    zeroOne.addElement("0");

    labels = new FastVector(2);
    labels.addElement("+");
    labels.addElement("-");
  }

  public void buildStumps(Instances rawTrainSet) throws Exception {
    for (int j = 0; j < 100; j++) {
      stumps[j] = new Id3();
      stumps[j].setMaxDepth(4);
      Instances curTrainSet = sampleHalfInstances(rawTrainSet);
      stumps[j].buildClassifier(curTrainSet);
    }
  }

  public Instances buildFeaturesFromStumps(Instances dataset) throws Exception {
    Instances instances = initializeAttributes();
    for (int i = 0; i < dataset.numInstances(); i++) {
      Instance rawInstance = dataset.instance(i);
      Instance newInstance = makeInstance(instances, rawInstance);
      instances.add(newInstance);
    }

    return instances;
  }

  protected Instances initializeAttributes() {

    String nameOfDataset = "Stumpped Badges";

    Instances instances;

    FastVector attributes = new FastVector();
    for (String featureName : features) {
      attributes.addElement(new Attribute(featureName, zeroOne));
    }
    Attribute classLabel = new Attribute("Class", labels);
    attributes.addElement(classLabel);

    instances = new Instances(nameOfDataset, attributes, 0);

    instances.setClass(classLabel);

    return instances;
  }

  protected Instance makeInstance(Instances newInstances, Instance rawInstance) throws Exception {
    Instance instance = new Instance(features.length + 1);
    instance.setDataset(newInstances);

    for (int i = 0; i < 100; i++) {
      Attribute att = newInstances.attribute(features[i]);
      int prediction = (int)stumps[i].classifyInstance(rawInstance);
      instance.setValue(att, String.valueOf(prediction));
    }

    instance.setClassValue(rawInstance.classValue());

    return instance;
  }

  protected Instances sampleHalfInstances(Instances rawTrainSet) {
    Instances resultSet = new Instances(rawTrainSet);
    Random randomGenerator = new Random();

    for (int i = 0; i < rawTrainSet.numInstances() / 2; i++) {
      int deleteIndex = randomGenerator.nextInt(resultSet.numInstances());
      resultSet.delete(deleteIndex);
    }
    return resultSet;
  }
}
