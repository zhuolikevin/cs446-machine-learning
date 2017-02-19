package cs446.homework2;

public class StatisticalUtil {
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
}
