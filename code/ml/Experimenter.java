package ml;

import ml.classifiers.*;
import ml.data.*;

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
     * Assess MultiLR Classifier performance with different iteration counts and learning rates
     * 
     * @param dataPath path to the wine dataset
     */
    public void assessLRPerformance(String dataPath) {
        DataSet dataset = new DataSet(dataPath, DataSet.TEXTFILE);

        //CrossValidationSet cvSet = dataset.getCrossValidationSet(10);
        //double[] learningRates = {0.001, 0.005, 0.01, 0.05, 0.1};
        double[] learningRates = {0.01};
        int[] iterations = {5, 10, 20, 50, 100};
        
        
        double bestAccuracy = 0.0;
        int bestIterations = 0;
        double bestLearningRate = 0.01;
        
        System.out.println("Step 1: Finding optimal iteration count (learning rate = 0.01)");
        System.out.println("----------------------------------------------------------------");
        
        for (int iter : iterations) {
            double accuracy = testLR(dataset, iter, 0.01);
            System.out.printf("%3d iterations, learning rate = 0.010, %.4f\n", iter, accuracy);
            
            if (accuracy > bestAccuracy) {
                bestAccuracy = accuracy;
                bestIterations = iter;
            }
        }
        bestIterations = 150;
        System.out.println();
        System.out.printf("Best iteration count: %d (accuracy: %.4f)\n", bestIterations, bestAccuracy);
        System.out.println();
        
        System.out.println("Step 2: Finding optimal learning rate (iterations = " + bestIterations + ")");
        System.out.println("----------------------------------------------------------------");
        
        for (int i = 0; i < 10; i++) {
        for (double lr : learningRates) {
            double accuracy = testLR(dataset, bestIterations, lr);
            System.out.printf("%3d iterations, learning rate = %.3f, %.4f\n", bestIterations, lr, accuracy);
            
            if (accuracy > bestAccuracy) {
                bestAccuracy = accuracy;
                bestLearningRate = lr;
            }
        }
    }
        
        System.out.println();
        System.out.printf("Best learning rate: %.3f (accuracy: %.4f)\n", bestLearningRate, bestAccuracy);
        System.out.println();
        
        System.out.println("Step 3: Fine-tuning with combinations around best parameters");
        System.out.println("--------------------------------------------------------------");
        
        int[] tuneIterations;
        if (bestIterations == iterations[0]) {
            tuneIterations = new int[]{bestIterations, bestIterations + 5};
        } else if (bestIterations == iterations[iterations.length - 1]) {
            tuneIterations = new int[]{bestIterations - 5, bestIterations};
        } else {
            tuneIterations = new int[]{bestIterations - 5, bestIterations, bestIterations + 5};
        }
        
        for (int iter : tuneIterations) {
            if (iter > 0 && iter <= 200) {
                double accuracy = testLR(dataset, iter, bestLearningRate);
                System.out.printf("%3d iterations, learning rate = %.3f, %.4f\n", iter, bestLearningRate, accuracy);
                
                if (accuracy > bestAccuracy) {
                    bestAccuracy = accuracy;
                    bestIterations = iter;
                }
            }
        }
        // Summary
        System.out.println();
        System.out.println("=== Best MultiLR Configuration ===");
        System.out.printf("  Iterations: %d\n", bestIterations);
        System.out.printf("  Learning Rate: %.3f\n", bestLearningRate);
        System.out.printf("  Accuracy: %.4f (%.2f%%)\n", bestAccuracy, bestAccuracy * 100);
        System.out.println();
    }
    
    /**
     * Compare Naive Bayes and MultiLR classifiers
     * 
     * @param dataPath path to the wine dataset
     */
    public void compareNaiveBayesAndLR(String dataPath) {
        DataSet dataset = new DataSet(dataPath, DataSet.TEXTFILE);
        CrossValidationSet cvSet = dataset.getCrossValidationSet(10);
        
        double bestNBAccuracy = 0.656;
        double bestLambda = 0.08;

        // Summary
        System.out.println();
        System.out.println("Best Naive Bayes Configuration (from Assignment 7):");
        System.out.printf("  Lambda: %.1f\n", bestLambda);
        System.out.printf("  Accuracy: %.4f (%.2f%%)\n", bestNBAccuracy, bestNBAccuracy * 100);
        System.out.println();
        
        System.out.println("Testing MultiLR Classifier");
        System.out.println("==================================================");
        int[] iterations = {1, 5, 10, 20, 50};
        
        double bestLRAccuracy = 0.0;
        int bestIterations = 0;
        for (int iter : iterations) {
            double accuracy = testLR(dataset, iter);
            System.out.printf("MultiLR with %2d iterations: %.4f\n", iter, accuracy);
            if (accuracy > bestLRAccuracy) {
                bestLRAccuracy = accuracy;
                bestIterations = iter;
            }
        }
        System.out.println();
        
        // Summary comparison
        System.out.println("=== Comparison Summary ===");
        System.out.println("==========================");
        System.out.printf("Naive Bayes: %.4f (%.2f%%)\n", bestNBAccuracy, bestNBAccuracy * 100);
        System.out.printf("  Best lambda: %.1f\n", bestLambda);
        System.out.printf("MultiLR: %.4f (%.2f%%)\n", bestLRAccuracy, bestLRAccuracy * 100);
        System.out.printf("  Best MultiLR iterations: %d\n", bestIterations);
        System.out.println();
    }
    
    /**
     * Test Multinomial Logistic Regression classifier with specified iterations (default learning rate)
     */
    private double testLR(DataSet dataset, int iterations) {
        return testLR(dataset, iterations, 0.01);
    }
    
    /**
     * Test MultiLRclassifier with specified iterations and learning rate
     */
    private double testLR(DataSet dataset, int iterations, double learningRate) {
        double totalCorrect = 0;
        double totalExamples = 0;
        
        for (int fold = 0; fold < 10; fold++) {
            DataSetSplit split = dataset.split(0.8);
            DataSet trainData = split.getTrain();
            DataSet testData = split.getTest();
            
            MultiLRClassifier classifier = new MultiLRClassifier();
            classifier.setIterations(iterations);
            classifier.setLearningRate(learningRate);
            classifier.train(trainData);
            
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
     * Test Naive Bayes classifier with specified lambda and feature usage
     * 
     * @param cvSet The cross validation set
     * @param lambda The smoothing parameter
     * @param useOnlyPositiveFeatures Whether to use only positive features or all features
     * @return The accuracy
     */
    private double testNaiveBayesWithLambda(CrossValidationSet cvSet, double lambda, boolean useOnlyPositiveFeatures) {
        double totalCorrect = 0;
        double totalExamples = 0;
        
        for (int fold = 0; fold < cvSet.getNumSplits(); fold++) {
            DataSetSplit split = cvSet.getValidationSet(fold);
            DataSet trainData = split.getTrain();
            DataSet testData = split.getTest();
            
            NBClassifier classifier = new NBClassifier();
            classifier.setLambda(lambda);
            classifier.setUseOnlyPositiveFeatures(useOnlyPositiveFeatures);
            classifier.train(trainData);
            
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
     * Main method to run the experiment
     */
    public static void main(String[] args) {
        Experimenter experimenter = new Experimenter();
        experimenter.runWineExperiment("/Users/pavelfilippov/IdeaProjects/assign-5-gradient-descent-pavel-filippov-and-tommy-liu/data/wines.train");
    }
}
