# CS446 Spring 2017 HW5

> Name: Zhuo Li

> NetID: zhuol2

> Email: zhuol2@illinois.edu

## Project Structure

- `parameter_tuning.py`: Parameter tuning for NN.
- `plot_learning_curve.py`: Plot learning curves for comparing NN and perceptron.

## Run the code

`matplotlib` is used for plot line charts with python, so we need to first get this library at head.

If you are using MacOS, just install with `pip`

```bash
$ pip install matplotlib
```

For other platforms, you can get instructions [here](http://matplotlib.org/users/installing.html)

### Parameter Tuning

```bash
$ python parameter_tuning.py
```

This will run parameter tuning for both `circles` and `mnist`

### Learning Curve

Comment out corresponding parts in `plot_learning_curve.py` for `circles` or `mnist`. Then run

```bash
$ python plot_learning_curve.py
```
