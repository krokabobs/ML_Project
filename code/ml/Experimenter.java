package ml;

import ml.classifiers.*;
import ml.data.*;

import java.util.ArrayList;

/**
 * Experimenter class to run the experiments from the assignment.
 * 
 * @author Pavel Filippov and Tommy Liu
 * Assignment 5
 */
public class Experimenter {
    /**
     * Performs 10-fold cross validation on wine dataset comparing OVA, AVA, and multiclass decision trees.
     * 
     * @param dataPath path to the wine dataset
     */
    public void runWineExperiment(String dataPath) {
        System.out.println("=== Wine Dataset 10-Fold Cross Validation Experiment ===");
        System.out.println("Dataset: " + dataPath);
        System.out.println();

        DataSet dataset = new DataSet(dataPath, DataSet.TEXTFILE);
        System.out.println("Loaded dataset with " + dataset.getData().size() + " examples");
        System.out.println("Number of classes: " + dataset.getLabels().size());
        System.out.println("Features: " + dataset.getAllFeatureIndices().size());
        System.out.println();

        // Create 10-fold cross validation
        CrossValidationSet cvSet = dataset.getCrossValidationSet(10);
        System.out.println("Created 10-fold cross validation");
        System.out.println();

        // Test configurations
        int[] depths = {1, 2, 3};
        int bestDepth = 19; // From previous analysis

        // Run experiments
        System.out.println("Testing OVA with Decision Trees (depths 1, 2, 3)");
        System.out.println("================================================");
        for (int depth : depths) {
            double accuracy = testOVA(cvSet, depth);
            System.out.printf("OVA Depth %d: %.4f (%.2f%%)\n", depth, accuracy, accuracy * 100);
        }
        System.out.println();

        System.out.println("Testing AVA with Decision Trees (depths 1, 2, 3)");
        System.out.println("================================================");
        for (int depth : depths) {
            double accuracy = testAVA(cvSet, depth);
            System.out.printf("AVA Depth %d: %.4f (%.2f%%)\n", depth, accuracy, accuracy * 100);
        }
        System.out.println();

        System.out.println("Testing Multiclass Decision Tree (best depth " + bestDepth + ")");
        System.out.println("=============================================================");
        double accuracy = testMulticlassDT(cvSet, bestDepth);
        System.out.printf("Multiclass DT Depth %d: %.4f (%.2f%%)\n", bestDepth, accuracy, accuracy * 100);
        System.out.println();

        // Summary table
        displaySummaryTable(cvSet, depths, bestDepth);
    }
    
    /**
     * Test OVA classifier with specified depth
     */
    private double testOVA(CrossValidationSet cvSet, int depth) {
        double totalCorrect = 0;
        double totalExamples = 0;
        
        for (int fold = 0; fold < cvSet.getNumSplits(); fold++) {
            DataSetSplit split = cvSet.getValidationSet(fold);
            DataSet trainData = split.getTrain();
            DataSet testData = split.getTest();
            
            // Create and train OVA classifier
            ClassifierFactory factory = new ClassifierFactory(ClassifierFactory.DECISION_TREE, depth);
            OVAClassifier classifier = new OVAClassifier(factory);
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
            ClassifierFactory factory = new ClassifierFactory(ClassifierFactory.DECISION_TREE, depth);
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
    
    /**
     * Test multiclass decision tree with specified depth
     */
    private double testMulticlassDT(CrossValidationSet cvSet, int depth) {
        double totalCorrect = 0;
        double totalExamples = 0;
        
        for (int fold = 0; fold < cvSet.getNumSplits(); fold++) {
            DataSetSplit split = cvSet.getValidationSet(fold);
            DataSet trainData = split.getTrain();
            DataSet testData = split.getTest();
            
            // Create and train multiclass decision tree
            DecisionTreeClassifier classifier = new DecisionTreeClassifier();
            classifier.setDepthLimit(depth);
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
    
    /**
     * Display summary table of results
     */
    private void displaySummaryTable(CrossValidationSet cvSet, int[] depths, int bestDepth) {
        System.out.println("Summary Table:");
        System.out.println("==============");
        System.out.printf("%-20s %-8s %-8s %-8s %-8s\n", "Method", "Depth 1", "Depth 2", "Depth 3", "Best");
        System.out.println("-------------------- -------- -------- -------- --------");
        
        // OVA results
        System.out.printf("%-20s", "OVA");
        double bestOVA = 0.0;
        for (int depth : depths) {
            double acc = testOVA(cvSet, depth);
            System.out.printf(" %-8.4f", acc);
            bestOVA = Math.max(bestOVA, acc);
        }
        System.out.printf(" %-8.4f\n", bestOVA);
        
        // AVA results
        System.out.printf("%-20s", "AVA");
        double bestAVA = 0.0;
        for (int depth : depths) {
            double acc = testAVA(cvSet, depth);
            System.out.printf(" %-8.4f", acc);
            bestAVA = Math.max(bestAVA, acc);
        }
        System.out.printf(" %-8.4f\n", bestAVA);
        
        // Multiclass DT
        double multiclassAcc = testMulticlassDT(cvSet, bestDepth);
        System.out.printf("%-20s %-8s %-8s %-8s %-8.4f\n", "Multiclass DT", "-", "-", "-", multiclassAcc);
        System.out.println();
        
        // Find overall best
        double overallBest = Math.max(bestOVA, Math.max(bestAVA, multiclassAcc));
        System.out.println("Overall Best Accuracy: " + String.format("%.4f", overallBest) + " (" + String.format("%.2f", overallBest * 100) + "%)");
    }
    
    /**
     * Main method to run the experiment
     */
    public static void main(String[] args) {
        Experimenter experimenter = new Experimenter();
        experimenter.runWineExperiment("/Users/pavelfilippov/IdeaProjects/assign-5-gradient-descent-pavel-filippov-and-tommy-liu/data/wines.train");
    }
}
