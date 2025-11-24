package ml.classifiers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Set;
import java.util.Random;

import ml.data.DataSet;
import ml.data.Example;

/**
 * Logistic Regression classifier
 * 
 * @author Pavel Filippov and Tommy Liu
 *
 */
public class LRClassifier implements Classifier {
	protected HashMap<Integer, Double> weights; // the feature weights
	protected double b = 0; // the intersect weight
	protected double alpha = 0.01; // learning rate
	protected int iterations = 10;
	

	/**
	 * Get a weight vector over the set of features with each weight
	 * set to 0
	 * 
	 * @param features the set of features to learn over
	 * @return
	 */
	protected HashMap<Integer, Double> getZeroWeights(Set<Integer> features){
		HashMap<Integer, Double> temp = new HashMap<Integer, Double>();
		
		for( Integer f: features){
			temp.put(f, 0.0);
		}
		
		return temp;
	}
	
	/**
	 * Initialize the weights and the intersect value
	 * 
	 * @param features
	 */
	protected void initializeWeights(Set<Integer> features){
		weights = getZeroWeights(features);
		b = 0;
	}

	/**
	 * Set the number of iterations the perceptron should run during training
	 * 
	 * @param iterations
	 */
	public void setIterations(int iterations){
		this.iterations = iterations;
	}

	public double dotProduct(Example example, HashMap<Integer, Double> weights){

		double dotProduct = 0.0;
		for (Integer feature : weights.keySet()){
			dotProduct += weights.get(feature) * example.getFeature(feature);
		}
		dotProduct += b;

		return dotProduct;
	}

	public double getPrediction(Example example){
		double dp = dotProduct(example, weights);
		return 1 / (1 + Math.exp(-dp));
	}
	
	public void train(DataSet data) {
		initializeWeights(data.getAllFeatureIndices());
		
		for (int iter = 0; iter < iterations; iter++){
			int exampleCount = 0;
			for (Example example : data.getData()){
				// System.out.println("Example" + exampleCount++);

				double prediction = getPrediction(example);
				// System.out.println("Prediction: " + prediction);

				double label = example.getLabel();
				double diff = prediction - label;
				// System.out.println("diff: " + diff);

				for (Integer feature : weights.keySet()){
					double oldWeight = weights.get(feature);
					double featureValue = example.getFeature(feature);
					double newWeight = oldWeight - alpha * diff * featureValue;
					// System.out.println("Feature " + feature + " old weight: " + oldWeight + " new weight: " + newWeight);
					weights.put(feature, newWeight);
				}
				b = b - alpha * diff;
				// System.out.println("New b: " + b);
			}
		}
	}

	@Override
	public double classify(Example example) {
		if (getPrediction(example) >= 0.5){
			return 1.0;
		} else {
			return -1.0;
		}
	}
	
	@Override
	public double confidence(Example example) {
		return Math.abs(getPrediction(example) - 0.5);
	}
}
