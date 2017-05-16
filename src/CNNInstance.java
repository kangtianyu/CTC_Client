import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.stream.Stream;

import org.neuroph.core.Connection;
import org.neuroph.core.Neuron;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.core.learning.error.MeanSquaredError;
import org.neuroph.nnet.comp.Kernel;
import org.neuroph.nnet.comp.layer.Layer2D;
import org.neuroph.nnet.learning.BackPropagation;

import net.sf.javaml.utils.ArrayUtils;

public class CNNInstance implements Runnable {

	class LearningListener implements LearningEventListener {


        long start = System.currentTimeMillis();
        int t_id;
		private Recorder re;
        
        public LearningListener(int t_id, Recorder re) {
			super();
			this.t_id = t_id;
			this.re = re;
		}

		@Override
        public void handleLearningEvent(LearningEvent event) {
            BackPropagation bp = (BackPropagation) event.getSource();
            if(re!=null){
            	re.count = bp.getCurrentIteration();
            	re.error = bp.getTotalNetworkError();
//            	re.l2.add(re.lambda*re.totWeightPenalty.getValue());
//            	System.out.println(re.l2);
            	if(re.hideIterMsg){
            		re.output+= String.format("Iter:%6d|Error:%10.7f%n",bp.getCurrentIteration(),bp.getTotalNetworkError());
            	}else{
                    System.out.println("Current iteration: " + bp.getCurrentIteration());
                    System.out.println("Error: " + bp.getTotalNetworkError());            		
            	}
            }else{
            	System.out.print("Itr: " + bp.getCurrentIteration());    
            	System.out.println("|Error: " + bp.getTotalNetworkError());
            }

//            System.out.println("Thread id:" + t_id);
//            System.out.println("Current iteration: " + bp.getCurrentIteration());
//            System.out.println("Error: " + bp.getTotalNetworkError());
//            System.out.println("Calculation time: " + (System.currentTimeMillis() - start) / 1000.0);

         //   neuralNetwork.save(bp.getCurrentIteration() + "CNN_MNIST" + bp.getCurrentIteration() + ".nnet");
            start = System.currentTimeMillis();
//            NeuralNetworkEvaluationService.completeEvaluation(neuralNetwork, testSet);
        }

    }
	
	private ArrayList<ArrayList<Double>> trainData;
	private ArrayList<Integer> groundTruth;
	private ArrayList<ArrayList<Double>> testData = new ArrayList<ArrayList<Double>>();
	private ArrayList<Integer> testgroundTruth = new ArrayList<Integer>();
	private ArrayList<ArrayList<Integer>> network = new ArrayList<ArrayList<Integer>>();
	private int inputSize;
	private int hiddenLayerSize;
	private StdDataset datasets;
	private boolean useSecondTrain = false;
	private Recorder re;

	public CNNInstance(
			ArrayList<ArrayList<Double>> trainData,
			ArrayList<Integer> groundTruth,
			ArrayList<ArrayList<Double>> testData,
			ArrayList<Integer> testgroundTruth,
			StdDataset datasets){
		this.trainData = trainData;
		this.groundTruth =groundTruth;
		this.testData = testData;
		this.testgroundTruth = testgroundTruth;
		this.network = datasets.getNetwork();
		this.hiddenLayerSize = datasets.getHiddenLayerSize();
		this.inputSize= trainData.get(0).size();
		this.datasets = datasets;
	}
	
	public ArrayList<Integer> getTestgroundTruth() {
		return testgroundTruth;
	}
	
	@Override
	public void run() {
		runtest();
	}
	
	public void runtest(){
		re = new Recorder();
		runtest(0,re);
	}
	
	public void runtest(int t_id, Recorder re){
		
		DataSet trainSet = new DataSet(inputSize,1);
		for(int i=0;i<trainData.size();i++){
			double[] input = Stream.of(trainData.get(i).toArray(new Double[0])).mapToDouble(Double::doubleValue).toArray();
			double[] output = {(double)groundTruth.get(i)};
//			System.out.println(i);
			trainSet.addRow(input, output);
		}
		
        Layer2D.Dimensions inputDimension = new Layer2D.Dimensions(inputSize, 1);
        Kernel convolutionKernel = new Kernel(5, 1);
		geneNetworkCNN convolutionNetwork = new geneNetworkCNN.Builder(inputDimension, 1,network)
        	.withConvolutionLayer(convolutionKernel, 1,hiddenLayerSize)
        	.withFullConnectedLayer(1)
        	.createNetwork();
//        JordanNetwork convolutionNetwork = new JordanNetwork(inputSize,20,20,1);
		
		if(re!=null){
			BackPropagation backPropagation = new L2MomentumBackpropagation(re.lambda,re.alpha);
	        backPropagation.setLearningRate(0.03);
	        backPropagation.setMaxError(0.0002);
	        backPropagation.setMaxIterations(20);
	        backPropagation.addListener(new LearningListener(t_id,re));
	        backPropagation.setErrorFunction(new MeanSquaredError());
	        convolutionNetwork.setLearningRule(backPropagation);
		}

//        backPropagation.addListener(new LearningListener());
              
//        System.out.println("Thread " + t_id + " started training...");  
		System.out.println("First learn");       
        convolutionNetwork.learn(trainSet);

        ObjectWithValue[] ary = new ObjectWithValue[convolutionNetwork.getLayers()[1].getNeuronsCount()];
        for(int i=0;i<convolutionNetwork.getLayers()[1].getNeuronsCount();i++){
        	ary[i] = new ObjectWithValue(i,convolutionNetwork.getLayers()[1].getNeuronAt(i).getOutConnections()[0].getWeight().value);
        }
//        Arrays.sort(ary);
        double threashold= Math.abs(ary[ary.length-100].value);
        Neuron outneu = convolutionNetwork.getOutputNeurons()[0];
        

        if(useSecondTrain ){
	        Connection[] cc = outneu.getInputConnections().clone();
	        for(Connection c:cc){
	        	if(Math.abs(c.getWeight().value)<threashold){
	        		outneu.removeInputConnectionFrom(c.getFromNeuron());
	        	}
	        }
	        
	        System.out.println("Second learn");       
	        convolutionNetwork.learn(trainSet);           
        }
//        System.out.println("Done training!");
        ArrayList<Double> outputs = new ArrayList<Double>();
        for(int i=0;i<testData.size();i++){
			double[] input = Stream.of(testData.get(i).toArray(new Double[0])).mapToDouble(Double::doubleValue).toArray();

			convolutionNetwork.setInput(input);
			convolutionNetwork.calculate();
			double result = convolutionNetwork.getOutput()[0];
			outputs.add(result);
        }
        re.outputs = outputs;
        
//        ary = new ObjectWithValue[convolutionNetwork.getLayers()[1].getNeuronsCount()];
//        for(int i=0;i<convolutionNetwork.getLayers()[1].getNeuronsCount();i++){
//        	ary[i] = new ObjectWithValue(i,convolutionNetwork.getLayers()[1].getNeuronAt(i).getOutConnections()[0].getWeight().value);
//        }
        ary = new ObjectWithValue[outneu.getInputConnections().length];
        int itr=0;
        for(Connection c:outneu.getInputConnections()){
        	ary[itr] = new ObjectWithValue(itr,c.getWeight().value);
        	itr++;
        }
        Arrays.sort(ary);
        re.ary =ary;
        
        re.println("------------");
        for(int i=ary.length-1;i>=ary.length-31;i--){
//            for(int i=ary.length-1;i>ary.length-11;i--){
        	re.println(datasets.getHiddenNodeContent((int)ary[i].o) + "," + ary[i]);
        }
        re.println("------------");
//        for(int i=100;i>=0;i--){
//        	re.println(datasets.getHiddenNodeContent((int)ary[i].o) + "," + ary[i]);
//        }
//        re.println("------------");
        
        
        double minDistance = 2;
        double useCutoff = 0;
        int TP = 0;
        int TN = 0;
        int FP = 0;
        int FN = 0;
        for(double cutoff=0;cutoff<1.001;cutoff+=0.001){
	        int mTP = 0;
	        int mTN = 0;
	        int mFP = 0;
	        int mFN = 0;
			
	        for(int i=0;i<testData.size();i++){
	        	double result = outputs.get(i);
	        	int output = testgroundTruth.get(i);

	//			double cutoff = 0.27;
				if(output == 1 && result >= cutoff)	mTP++;
				if(output == 1 && result < cutoff)	mFN++;
				if(output == 0 && result >= cutoff)	mFP++;
				if(output == 0 && result < cutoff)	mTN++;
			}
			double tpr = ((double)mTP)/(mTP+mFN);
			double fpr = ((double)mFP)/(mFP+mTN);
			double distance = Math.sqrt((1-tpr)*(1-tpr)+fpr*fpr);
			if(distance < minDistance){
				TP = mTP;
				TN = mTN;
				FP = mFP;
				FN = mFN;
				minDistance = distance;
				useCutoff = cutoff;
			}
			re.cutoff.add(cutoff);
			re.tpr.add(tpr);
			re.fpr.add(fpr);
        }
        ArrayList<Integer> predict = new ArrayList<Integer>();
        double error = 0.0;
        for(int i=0;i<testData.size();i++){
        	double result = outputs.get(i);
        	int output = testgroundTruth.get(i);
        	error += (result-output)*(result-output);
	        if(result>=useCutoff){
				predict.add(1);
			}else{
				predict.add(0);
			}        	
        }
        int tot = TP+TN+FP+FN;
		DecimalFormat numberFormat = new DecimalFormat("0.0000");
		if(re!=null){
			re.println("************Thread id is " + t_id + " ************"+numberFormat.format(useCutoff));
			re.println("\tpredict+\tpredict-\tPrevalence= " + numberFormat.format(((double)TP+FN)/tot));
			re.println("+\t" + TP +"("+numberFormat.format(((double)TP)/tot)+")\t" + 
					FN +"("+numberFormat.format(((double)FN)/tot)+
					")\tSensitivity="+numberFormat.format(((double)TP)/(TP+FN))+
					"\tMiss rate=  "+numberFormat.format(((double)FN)/(TP+FN)));
			re.println("-\t" + FP +"("+numberFormat.format(((double)FP)/tot)+")\t" +
					TN +"("+numberFormat.format(((double)TN)/tot)+
					")\tFall-out=   "+numberFormat.format(((double)FP)/(FP+TN))+
					"\tSpecificity="+numberFormat.format(((double)TN)/(FP+TN)));
			re.println("\t\t"+
					"\tAccuracy="+numberFormat.format(((double)TP+TN)/tot)+
					"\tPrecision=  "+numberFormat.format(((double)TP)/(TP+FP)));
			re.println("***************************************");
			re.predict = predict;
			re.error = error;
			re.sensitivity = ((double)TP)/(TP+FN);
			re.specificity = ((double)TN)/(FP+TN);
			re.end = true;
		}else{
			System.out.println("************Thread id is " + t_id + " ************");
			System.out.println("\tpredict+\tpredict-\tPrevalence= " + numberFormat.format(((double)TP+FN)/tot));
			System.out.println("+\t" + TP +"("+numberFormat.format(((double)TP)/tot)+")\t" + 
					FN +"("+numberFormat.format(((double)FN)/tot)+
					")\tSensitivity="+numberFormat.format(((double)TP)/(TP+FN))+
					"\tMiss rate=  "+numberFormat.format(((double)FN)/(TP+FN)));
			System.out.println("-\t" + FP +"("+numberFormat.format(((double)FP)/tot)+")\t" +
					TN +"("+numberFormat.format(((double)TN)/tot)+
					")\tFall-out=   "+numberFormat.format(((double)FP)/(FP+TN))+
					"\tSpecificity="+numberFormat.format(((double)TN)/(FP+TN)));
			System.out.println("\t\t"+
					"\tAccuracy="+numberFormat.format(((double)TP+TN)/tot)+
					"\tPrecision=  "+numberFormat.format(((double)TP)/(TP+FP)));
			System.out.println("***************************************");
		}
	}

	public Recorder getRecorder() {
		return re;
	}

}
