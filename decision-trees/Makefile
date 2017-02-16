JFLAGS = -cp lib/weka.jar:bin -d bin -sourcepath src
JAVAC = javac

.PHONY: directories
.SUFFIXES: .java .class

.java.class:
	$(JAVAC) $(JFLAGS) $*.java

CLASSES = src/StatisticalUtil.java src/FeatureGenerator.java src/Id3.java src/WekaTester.java src/SGD.java src/SGDTester.java src/StumpSGD.java src/StumpSGDTester.java

all: directories classes

directories: 
	mkdir -p bin

classes: $(CLASSES:.java=.class)

clean:
	$(RM) -r bin

