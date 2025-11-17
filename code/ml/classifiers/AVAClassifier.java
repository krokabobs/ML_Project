package ml.classifiers;

import ml.data.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Set;
import java.util.Map;

/**
 * All-Versus-All classifier that trains a binary classifier for each pair of classes,
 * then uses weighted voting based on confidence to classify examples.
 * 
 * @author Pavel Filippov and Tommy Liu
 * Assignment 5
 */
public class AVAClassifier implements Classifier {
    private ClassifierFactory factory;
    private ArrayList<Classifier> classifiers;
    
    private ArrayList<Double> class1Labels;
    private ArrayList<Double> class2Labels;
    private boolean trained = false;
    private ArrayList<Double> datasetLabels;
    
    /**
     * Constructor for AVAClassifier.
     * 
     * @param classifierFactory the factory to create individual binary classifiers
     */
    public AVAClassifier(ClassifierFactory classifierFactory) {
        this.factory = classifierFactory;
        this.classifiers = new ArrayList<Classifier>();
        this.class1Labels = new ArrayList<Double>();
        this.class2Labels = new ArrayList<Double>();
        this.datasetLabels = new ArrayList<Double>();
    }

    /**
     * Create a binary dataset for a pair of classes.
     * 
     * @param originalData the original multiclass dataset
     * @param positiveClass the first class (treated as positive)
     * @param negativeClass the second class (treated as negative)
     * @return binary dataset for the pair classification
     */
    private DataSet createBinaryDataset(DataSet originalData, Double positiveClass, Double negativeClass) {
        DataSet binaryData = new DataSet(originalData.getFeatureMap());
        
        for (Example example : originalData.getData()) {
            double label = example.getLabel();
            
            // Only include examples from the two classes we're comparing
            if (Math.abs(label - positiveClass) < 0.001) {
                Example binaryExample = new Example(example);
                binaryExample.setLabel(1.0); // positive class
                binaryData.addData(binaryExample);
            } else if (Math.abs(label - negativeClass) < 0.001) {
                Example binaryExample = new Example(example);
                binaryExample.setLabel(-1.0); // negative class
                binaryData.addData(binaryExample);
            }
            // Skip examples not belonging to either class (no continue needed)
        }
        
        return binaryData;
    }
    
    /**
     * Train the AVA classifier on the given dataset.
     * 
     * @param dataset the dataset to train on
     */
    public void train(DataSet data) {
        // Get all unique class labels
        Set<Double> labels = data.getLabels();
        this.datasetLabels = new ArrayList<Double>(labels);
        class1Labels.clear();
        class2Labels.clear();
        classifiers.clear();
        
        // Create a binary classifier for each class
        for (int i = 0; i < datasetLabels.size(); i++) {
            for (int j = i + 1 ; j < datasetLabels.size(); j++) {
                Double label1 = datasetLabels.get(i);
                Double label2 = datasetLabels.get(j);
                class1Labels.add(label1);
                class2Labels.add(label2);
                
                // Create binary dataset for this class vs all others
                DataSet binaryData = createBinaryDataset(data, label1, label2);
                
                // Create and train classifier for this pair
                Classifier binaryClassifier = factory.getClassifier();
                binaryClassifier.train(binaryData);
                classifiers.add(binaryClassifier);
            }
        }
        
        trained = true;
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
        
        // Initialize vote totals for all classes
        Map<Double, Double> votes = new HashMap<>();
        for (Double x : datasetLabels){
            votes.put(x, 0.0);
        }

        // For each binary classifier, update vote totals
        for (int i = 0; i < classifiers.size(); i++) {
            Classifier classifier = classifiers.get(i);
            double confidence = classifier.confidence(example);
            
            // Get prediction from this classifier
            Example binaryExample = new Example(example);
            binaryExample.setLabel(0.0); // set dummy label for classification
            double prediction = classifier.classify(binaryExample);
            
            Double class1 = class1Labels.get(i);
            Double class2 = class2Labels.get(i);
            
            if (Math.abs(prediction - 1.0) < 0.001) {
                // Classifier predicts class1 (positive)
                votes.put(class1, votes.get(class1) + confidence);
                votes.put(class2, votes.get(class2) - confidence);
            } else if (Math.abs(prediction + 1.0) < 0.001) {
                // Classifier predicts class2 (negative)
                votes.put(class1, votes.get(class1) - confidence);
                votes.put(class2, votes.get(class2) + confidence);
            }
        }

        // Find the class with the highest vote total
        double bestLabel = datasetLabels.get(0); // default to first class
        double bestVote = votes.get(bestLabel);
        
        for (Double label : datasetLabels) {
            if (votes.get(label) > bestVote) {
                bestVote = votes.get(label);
                bestLabel = label;
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