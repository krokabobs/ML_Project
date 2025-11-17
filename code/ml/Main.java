package ml;

import ml.classifiers.*;
import ml.data.*;
import ml.utils.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
import ml.Experimenter;

public class Main {
    /**
     * Test AVA classifier with specified depth
     */
    private double testAVA(CrossValidationSet cvSet, int depth) {
        double totalCorrect = 0;
        double totalExamples = 0;
        
        for (int fold = 0; fold < cvSet.getNumSplits(); fold++) {
            DataSetSplit split = cvSet.getValidationSet(fold);
            DataSet trainData = split.getTrain();
            DataSet testData = split.getTest();
            
            // Create and train AVA classifier
            ClassifierFactory factory = new ClassifierFactory(ClassifierFactory.LOGISTIC_REGRESSION, depth);
            AVAClassifier classifier = new AVAClassifier(factory);
            classifier.train(trainData);
            
            // Test on this fold
            int correct = 0;
            for (Example example : testData.getData()) {
                double prediction = classifier.classify(example);
                if (Math.abs(prediction - example.getLabel()) < 0.001) {
                    correct++;
                }
            }
            
            totalCorrect += correct;
            totalExamples += testData.getData().size();
        }
        
        return totalCorrect / totalExamples;
    }
    
    public static void main(String[] args) {
    
        // Experimenter experimenter = new Experimenter();
        // experimenter.runWineExperiment("data/wines.train");
        // DataSet wineDataset = new DataSet("data/wines.trainSample", DataSet.TEXTFILE);
        // LRClassifier classifier = new LRClassifier();
        // classifier.setIterations(10);



        // ClassifierFactory factory = new ClassifierFactory(ClassifierFactory.LOGISTIC_REGRESSION, 3);
        // OVAClassifier classifier = new OVAClassifier(factory);


        DataSet wineDataset = new DataSet("data/default.csv", DataSet.CSVFILE);
        LRClassifier classifier = new LRClassifier();

        classifier.setIterations(1);
        classifier.train(wineDataset);
        int correct = 0;
        for (Example example : wineDataset.getData()) {
            double prediction = classifier.classify(example);
            if (Math.abs(prediction - example.getLabel()) < 0.001) {
                correct++;
            }
        }
        System.out.println("Accuracy: " + ((double) correct / wineDataset.getData().size()));

        classifier.setIterations(1);
        classifier.train(wineDataset);
    }
}