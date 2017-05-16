import java.util.ArrayList;

import org.neuroph.core.Layer;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.Neuron;
import org.neuroph.core.exceptions.VectorSizeMismatchException;
import org.neuroph.core.input.WeightedSum;
import org.neuroph.nnet.comp.Kernel;
import org.neuroph.nnet.comp.layer.FeatureMapsLayer;
import org.neuroph.nnet.comp.layer.InputMapsLayer;
import org.neuroph.nnet.comp.layer.Layer2D;
import org.neuroph.nnet.comp.layer.PoolingLayer;
import org.neuroph.nnet.comp.neuron.BiasNeuron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.ConnectionFactory;
import org.neuroph.util.NeuronProperties;
import org.neuroph.util.TransferFunctionType;


public class geneNetworkCNN extends NeuralNetwork<BackPropagation> {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -821717403754358444L;

	public geneNetworkCNN(){
		
	}
	
	
    @Override
    public void setInput(double... inputVector) throws VectorSizeMismatchException {
        FeatureMapsLayer inputLayer = (FeatureMapsLayer) getLayerAt(0);
        int currentNeuron = 0;
        for (int i = 0; i < inputLayer.getNumberOfMaps(); i++) {
            Layer2D map = inputLayer.getFeatureMap(i);
            for (Neuron neuron : map.getNeurons()) {
                if (!(neuron instanceof BiasNeuron))
                    neuron.setInput(inputVector[currentNeuron++]);
            }
        }
    }

    public static class Builder {

        public static final NeuronProperties DEFAULT_FULL_CONNECTED_NEURON_PROPERTIES = new NeuronProperties();
        private geneNetworkCNN network;
		private ArrayList<ArrayList<Integer>> geneNetwork;

        static {
            DEFAULT_FULL_CONNECTED_NEURON_PROPERTIES.setProperty("useBias", true);
            DEFAULT_FULL_CONNECTED_NEURON_PROPERTIES.setProperty("transferFunction", TransferFunctionType.SIGMOID);
            DEFAULT_FULL_CONNECTED_NEURON_PROPERTIES.setProperty("inputFunction", WeightedSum.class);
        }

        public Builder(Layer2D.Dimensions mapSize, int numberOfMaps,
    			ArrayList<ArrayList<Integer>> geneNetwork) {
            network = new geneNetworkCNN();
            InputMapsLayer inputLayer = new InputMapsLayer(mapSize, numberOfMaps);
            inputLayer.setLabel("Input Layer");
            network.addLayer(inputLayer);
            this.geneNetwork = geneNetwork;

        }

        public Builder withConvolutionLayer(final Kernel convolutionKernel, int numberOfMaps, int hiddenLayerSize) {
            FeatureMapsLayer lastLayer = getLastFeatureMapLayer();
            geneNetworkCNNLayer convolutionLayer = new geneNetworkCNNLayer(lastLayer, convolutionKernel, numberOfMaps,geneNetwork,hiddenLayerSize);

            network.addLayer(convolutionLayer);
            geneNetworkCNNUtils.fullConnectMapLayers(lastLayer, convolutionLayer);

            return this;
        }

        public Builder withPoolingLayer(final Kernel poolingKernel) {
            FeatureMapsLayer lastLayer = getLastFeatureMapLayer();
            PoolingLayer poolingLayer = new PoolingLayer(lastLayer, poolingKernel);

            network.addLayer(poolingLayer);
            geneNetworkCNNUtils.fullConnectMapLayers(lastLayer, poolingLayer);

            return this;
        }

        public Builder withFullConnectedLayer(int numberOfNeurons) {
            Layer lastLayer = getLastLayer();

            Layer fullConnectedLayer = new Layer(numberOfNeurons, DEFAULT_FULL_CONNECTED_NEURON_PROPERTIES);
            network.addLayer(fullConnectedLayer);

            ConnectionFactory.fullConnect(lastLayer, fullConnectedLayer);

            return this;
        }

        public geneNetworkCNN createNetwork() {
            network.setInputNeurons(network.getLayerAt(0).getNeurons());
            network.setOutputNeurons(getLastLayer().getNeurons());
            network.setLearningRule(new MomentumBackpropagation());
            return network;
        }


        private FeatureMapsLayer getLastFeatureMapLayer() {
            Layer layer = getLastLayer();
            if (layer instanceof FeatureMapsLayer)
                return (FeatureMapsLayer) layer;

            throw new RuntimeException("Unable to add next layer because previous layer is not FeatureMapLayer");
        }

        private Layer getLastLayer() {
            return network.getLayerAt(network.getLayersCount() - 1);
        }


    }
}
