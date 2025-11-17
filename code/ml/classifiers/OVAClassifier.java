package ml.classifiers;

import ml.data.*;
import java.util.ArrayList;
import java.util.Set;

/**
 * One-Versus-All classifier that trains a binary classifier for each class
 * against all other classes, then selects the most confident positive prediction.
 * 
 * @author Pavel Filippov and Tommy Liu
 * Assignment 5
 */
public class OVAClassifier implements Classifier {
    private ClassifierFactory factory;
    private ArrayList<Classifier> classifiers;
    private ArrayList<Double> classLabels;
    private boolean trained = false;
    
    /**
     * Constructor for OVAClassifier.
     * 
     * @param classifierFactory the factory to create individual binary classifiers
     */
    public OVAClassifier(ClassifierFactory classifierFactory) {
        this.factory = classifierFactory;
        this.classifiers = new ArrayList<Classifier>();
        this.classLabels = new ArrayList<Double>();
    }
    
    /**
     * Get the list of binary classifiers.
     * 
     * @return list of classifiers
     */
    public ArrayList<Classifier> getClassifiers() {
        return classifiers;
    }

    /**
     * Get the list of class labels.
     * 
     * @return list of class labels
     */
    public ArrayList<Double> getClassLabels() {
        return classLabels;
    }

    /**
     * Train the OVA classifier by creating one binary classifier per class.
     * 
     * @param data the training dataset
     */
    public void train(DataSet data) {
        // Get all unique class labels
        Set<Double> labels = data.getLabels();
        classLabels.clear();
        classifiers.clear();
        
        // Create a binary classifier for each class
        for (Double label : labels) {
            classLabels.add(label);
            
            // Create binary dataset for this class vs all others
            DataSet binaryData = createBinaryDataset(data, label);
            
            // Create and train classifier for this class
            Classifier binaryClassifier = factory.getClassifier();
            binaryClassifier.train(binaryData);
            classifiers.add(binaryClassifier);
        }
        
        trained = true;
    }
    
    /**
     * Create a binary dataset where the target class is positive (1.0) and all others are negative (0.0).
     * 
     * @param originalData the original multiclass dataset
     * @param positiveClass the class to treat as positive
     * @return binary dataset for one-vs-all classification
     */
    private DataSet createBinaryDataset(DataSet originalData, Double positiveClass) {
        DataSet binaryData = new DataSet(originalData.getFeatureMap());
        
        for (Example example : originalData.getData()) {
            Example binaryExample = new Example(example);
            if (Math.abs(example.getLabel() - positiveClass) < 0.001) {
                binaryExample.setLabel(1.0); // positive class
            } else {
                binaryExample.setLabel(-1.0); // negative class
            }
            binaryData.addData(binaryExample);
        }
        
        return binaryData;
    }
    
    /**
     * Classify an example using the most confident positive prediction.
     * 
     * @param example the example to classify
     * @return the predicted class label
     */
    public double classify(Example example) {
        if (!trained) {
            throw new RuntimeException("Classifier has not been trained yet");
        }
        
        double bestConfidence = Double.NEGATIVE_INFINITY;
        double bestLabel = classLabels.get(0); // default to first class
        boolean foundPositive = false;
        
        // Find the most confident positive prediction
        for (int i = 0; i < classifiers.size(); i++) {
            Classifier classifier = classifiers.get(i);
            double confidence = classifier.confidence(example);
            
            // // Check if this classifier predicts positive (class 1.0)
            // Example binaryExample = new Example(example);
            // binaryExample.setLabel(0.0); // set dummy label for confidence calculation
            // double prediction = classifier.classify(binaryExample);


            // Check if this classifier predicts positive (class 1.0)
            double prediction = classifier.classify(example);
            
            if (Math.abs(prediction - 1.0) < 0.001) { // this classifier predicts positive
                foundPositive = true;
                if (confidence > bestConfidence) {
                    bestConfidence = confidence;
                    bestLabel = classLabels.get(i);
                }
            }
        }
        
        // If no classifier predicted positive, return the least confident prediction
        if (!foundPositive) {
            double worstConfidence = Double.POSITIVE_INFINITY;
            for (int i = 0; i < classifiers.size(); i++) {
                Classifier classifier = classifiers.get(i);
                double confidence = classifier.confidence(example);
                if (confidence < worstConfidence) {
                    worstConfidence = confidence;
                    bestLabel = classLabels.get(i);
                }
            }
        }
        
        return bestLabel;
    }
    
    /**
     * Return confidence for the classification.
     * 
     * @param example the example to get confidence for
     * @return 0 a placeholder
     */
    public double confidence(Example example) {
        return 0.0;
    }
}