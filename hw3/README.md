# CS446 Spring 2017 HW3

> Name: Zhuo Li

> NetID: zhuol2

> Email: zhuol2@illinois.edu

## Project Structure

- `algorithms/`: This directory contains the implementations of three algorithms (Perceptron, Winnow and AdaGrad).
- `results/`: Some console outputs recorded during the experiments.
- `data_generator.py`: Common scripts for data generation.
- `tune_parameters.py`: General tuning methods.
- Files for each specific experiment are named with numbers. e.g. prefix `exp1_` stands for codes for the first experiment.

## Run the code

### Preparation

`matplotlib` is used for plot line charts with python, so we need to first get this library at head.

If you are using MacOS, just install with `pip`

```bash
$ pip install matplotlib
```

For other platforms, you can get instructions [here](http://matplotlib.org/users/installing.html)

### Experiment 1

- Generate required data

  ```bash
  $ ./exp1_generate_data.sh
  ```

- Parameters Tuning

  ```bash
  $ python exp1_tune_parameters.py
  ```

- Plot graphs

  ```bash
  $ python exp1_plot.py
  ```

### Experiment 2

- Generate required data

  ```bash
  $ ./exp2_generate_data.sh
  ```

- Parameters Tuning

  ```bash
  $ python exp2_tune_parameters.py
  ```

- Plot graphs

  ```bash
  $ python exp2_plot.py
  ```

### Experiment 3

- Generate required data

  ```bash
  $ ./exp3_generate_data.sh
  ```

  Note that some extra data may be generated but never used. See the file comments for more.

- Parameters Tuning

  ```bash
  $ python exp3_tune_parameters.py
  ```

- Get accuracies for different configurations per algorithms

  ```bash
  $ python exp3_train_evaluation.py
  ```

### Experiment 4 (Bonus)

- Generate required data

  ```bash
  $ ./exp4_generate_data.sh
  ```

- Plot graphs

  ```bash
  $ python exp4_plot.py
  ```
