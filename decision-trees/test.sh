#!/bin/bash

mkdir bin

make

#for i in {1..5}
#do
#  java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badges/badges.modified.data.fold$i ./../badges.fold$i.arff
#done
#
#java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badges/badges.modified.data.fold2345 ./../badges.fold2345.arff
#java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badges/badges.modified.data.fold1345 ./../badges.fold1345.arff
#java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badges/badges.modified.data.fold1245 ./../badges.fold1245.arff
#java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badges/badges.modified.data.fold1235 ./../badges.fold1235.arff
#java -cp lib/weka.jar:bin cs446.homework2.FeatureGenerator ../badges/badges.modified.data.fold1234 ./../badges.fold1234.arff

# SGD
#java -cp lib/weka.jar:bin cs446.homework2.SGDTester ./../badges.fold5.arff ./../badges.fold4.arff ./../badges.fold3.arff ./../badges.fold2.arff ./../badges.fold1.arff ./../badges.fold1234.arff ./../badges.fold1235.arff ./../badges.fold1245.arff ./../badges.fold1345.arff ./../badges.fold2345.arff

# Decision Trees with unlimited, 4 and 8 depth
java -cp lib/weka.jar:bin cs446.homework2.WekaTester ./../badges.fold5.arff ./../badges.fold4.arff ./../badges.fold3.arff ./../badges.fold2.arff ./../badges.fold1.arff ./../badges.fold1234.arff ./../badges.fold1235.arff ./../badges.fold1245.arff ./../badges.fold1345.arff ./../badges.fold2345.arff

# SGD with decision stump as features
#java -cp lib/weka.jar:bin cs446.homework2.StumpSGDTester ./../badges.fold5.arff ./../badges.fold4.arff ./../badges.fold3.arff ./../badges.fold2.arff ./../badges.fold1.arff ./../badges.fold1234.arff ./../badges.fold1235.arff ./../badges.fold1245.arff ./../badges.fold1345.arff ./../badges.fold2345.arff
