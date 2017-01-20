import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

/**
 * Your implementation of a naive bayes classifier. Please implement all four methods.
 */

public class NaiveBayesClassifierImpl implements NaiveBayesClassifier {
	/**
	 * Trains the classifier with the provided training data and vocabulary size
	 */
	private Map<String, ArrayList<Integer>> map = new HashMap<String, ArrayList<Integer>>();
	private int spamInstances;
	private int hamInstances;
	private int v;
	private int spamWords;
	private int hamWords;
	
	@Override
	public void train(Instance[] trainingData, int v) {
		// Implement		
		this.v = v;
		for(int i = 0; i < trainingData.length;i++){
			if(trainingData[i].label == Label.SPAM) {
				spamInstances++;
			}else{
				hamInstances++;
			}
			for(int j = 0; j<trainingData[i].words.length;j++){
				if(map.containsKey(trainingData[i].words[j])){
					ArrayList<Integer> count1 = map.get(trainingData[i].words[j]);
					if(trainingData[i].label == Label.SPAM){
						spamWords++;
						count1.set(0, count1.get(0)+1);
					}else{
						hamWords++;
						count1.set(1,count1.get(1)+1);
					}
				}else{
					ArrayList<Integer> count2 = new ArrayList<Integer>();
					count2.add(0);
					count2.add(0);
					if(trainingData[i].label == Label.SPAM){
						spamWords++;
						count2.set(0, count2.get(0)+1);
					}else{
						hamWords++;
						count2.set(1, count2.get(1)+1);
					}
					map.put(trainingData[i].words[j], count2);
				}
					
			}
			
		}
	}

	/**
	 * Returns the prior probability of the label parameter, i.e. P(SPAM) or P(HAM)
	 */
	@Override
	public double p_l(Label label) {
		// Implement
		double p1 = (double)spamInstances / (hamInstances + spamInstances);
		double p2 = (double)hamInstances / (hamInstances + spamInstances);
		
		if(label==Label.SPAM){
			return p1;
		}else{
			return p2;
		}
		//return 0;
	}

	/**
	 * Returns the smoothed conditional probability of the word given the label,
	 * i.e. P(word|SPAM) or P(word|HAM)
	 */
	@Override
	public double p_w_given_l(String word, Label label) {
		// Implement
		int i = 0;
		int count = 0;
		double condtionalP = 0.0;
		double o = 0.00001;
		if(label == Label.SPAM) {
			i = 0;
		}else{
			i = 1;
		}
		if(map.containsKey(word)){
			count = map.get(word).get(i);
		}else{
			count = 0;
		}
		////////////////////////////////////////////////////////////////////////////////
		condtionalP = (((double)count) + o)/ (v*o+  ((i == 0) ? spamWords : hamWords));
		
		return condtionalP;
	}
	
	/**
	 * Classifies an array of words as either SPAM or HAM. 
	 */
	@Override
	public ClassifyResult classify(String[] words) {
		// Implement
		ClassifyResult result = new ClassifyResult();
		double logSpam = 0.0;
		double logHam = 0.0;
		for(int i =0; i < words.length; i++){
			logSpam += Math.log(p_w_given_l(words[i], Label.SPAM));
			logHam += Math.log(p_w_given_l(words[i], Label.HAM));
		}
		result.log_prob_ham = Math.log(p_l(Label.HAM))+ logHam;
		result.log_prob_spam = Math.log(p_l(Label.SPAM))+ logSpam;
		if(result.log_prob_spam < result.log_prob_ham){
			result.label = Label.HAM;
		}else{
			result.label = Label.SPAM;
		}
		return result;
		//return null;
	}
}


