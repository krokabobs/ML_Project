package ml.classifiers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Set;
import java.util.Random;

import ml.data.DataSet;
import ml.data.Example;

/**
 * Multinomial Logistic Regression classifier (Softmax Regression)
 *
 * (We acknowledge that the code below was written by ChatGPT due to the time constraint, but we did go over
 * the code line by line and verified that it was indeed what we wanted.)
 * 
 * We wrote the LR Classifier code by ourselves.
 */
public class MultiLRClassifier implements Classifier {

    protected double[][] W;          // W[class][feature]
    protected double[] b;            // bias per class
    protected double alpha = 0.01;   // learning rate
    protected int iterations = 10;

    private int numClasses;
    private int numFeatures;

    /** 
     * Set the number of iterations for training 
     */
    public void setIterations(int iterations) {
        this.iterations = iterations;
    }

    /**
	 * Set the learning rate (alpha) for gradient descent
	 * 
	 * @param alpha the learning rate
	 */
	public void setLearningRate(double alpha){
		this.alpha = alpha;
	}

    /** Softmax function */
    private double[] softmax(double[] z) {
        double max = z[0];
        for (double v : z) max = Math.max(max, v);

        double sum = 0.0;
        double[] expZ = new double[z.length];
        for (int i = 0; i < z.length; i++) {
            expZ[i] = Math.exp(z[i] - max);
            sum += expZ[i];
        }
        for (int i = 0; i < z.length; i++) {
            expZ[i] /= sum;
        }
        return expZ;
    }

    /** Dot product with class c */
    private double dot(Example e, int c) {
        double dp = b[c];
        for (int f : e.getFeatureSet()) {
            dp += W[c][f] * e.getFeature(f);
        }
        return dp;
    }

    /** Predict class label (argmax probability) */
    @Override
    public double classify(Example e) {
        double[] logits = new double[numClasses];
        for (int c = 0; c < numClasses; c++) {
            logits[c] = dot(e, c);
        }
        double[] probs = softmax(logits);

        int best = 0;
        for (int c = 1; c < numClasses; c++) {
            if (probs[c] > probs[best]) best = c;
        }
        return best;
    }

    /** Confidence = probability of predicted class */
    @Override
    public double confidence(Example e) {
        double[] logits = new double[numClasses];
        for (int c = 0; c < numClasses; c++) {
            logits[c] = dot(e, c);
        }
        double[] probs = softmax(logits);

        int best = 0;
        for (int c = 1; c < numClasses; c++) {
            if (probs[c] > probs[best]) best = c;
        }
        return probs[best];
    }

    /** Multiclass training */
    @Override
    public void train(DataSet data) {

        // CLASS / FEATURE SETUP
        numClasses = data.getLabels().size();
        numFeatures = data.getAllFeatureIndices().size();

        W = new double[numClasses][numFeatures];
        b = new double[numClasses];

        // TRAINING LOOP
        for (int iter = 0; iter < iterations; iter++) {

            for (Example e : data.getData()) {

                int y = (int) e.getLabel();    // true label

                // Compute logits
                double[] logits = new double[numClasses];
                for (int c = 0; c < numClasses; c++) {
                    logits[c] = dot(e, c);
                }

                // Compute softmax probabilities
                double[] probs = softmax(logits);

                // GRADIENT UPDATE
                for (int c = 0; c < numClasses; c++) {
                    double error = probs[c] - (c == y ? 1.0 : 0.0);

                    // Update weights
                    for (int f : e.getFeatureSet()) {
                        double x = e.getFeature(f);
                        W[c][f] -= alpha * error * x;
                    }

                    // Update bias
                    b[c] -= alpha * error;
                }
            }
        }
    }
}
