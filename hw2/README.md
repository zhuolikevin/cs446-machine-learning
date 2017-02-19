# CS446 Spring 2017 HW2

> Name: Zhuo Li
> NetID: zhuol2
> Email: zhuol2@illinois.edu

## Directory Descriptions

Despite the original files, the following files are added/modified:

- Concatenated data sets for CV
  - `badges.modified.data.fold2345`
  - `badges.modified.data.fold1345`
  - `badges.modified.data.fold1245`
  - `badges.modified.data.fold1235`
  - `badges.modified.data.fold1234`

- tree files
  - `tree_unlimited`: the unlimited tree with best performance among folds
  - `tree_depth_4`: the depth-4 tree with best performance among folds
  - `tree_depth_8`: the depth-8 tree with best performance among folds

- `test.sh`: this file will compiler all the java source codes and run for 5 algorithms

- java source codes
  - `src/FeatureGenerator.java`: generate 260 features as the problem mentioned
  - `src/SGD.java`: SGD classifier
  - `src/SGDTester.java`: Train and test with SGD
  - `src/StumpSGD.java`: SGD with decision stump powered features
  - `src/StumpSGDTester.java`: Train and test with stumped SGD
  - `src/WekaTester.java`: Train and test with DT including unlimited, depth of 4 and depth of 8
  - `src/StatisticalUtil.java`: Statistical util functions

## Run

Simply use the `test.sh` script for running all the 5 algorithms

```bash
$ ./test.sh
```

If you want to run a specific algorithm, comment out other algorithms at the bottom of `test.sh`.
