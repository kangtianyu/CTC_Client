import java.util.ArrayList;
import org.neuroph.core.Neuron;
import org.neuroph.core.Weight;
import org.neuroph.core.input.WeightedSum;
import org.neuroph.core.transfer.RectifiedLinear;
import org.neuroph.core.transfer.Tanh;
import org.neuroph.nnet.comp.Kernel;
import org.neuroph.nnet.comp.layer.ConvolutionalLayer;
import org.neuroph.nnet.comp.layer.FeatureMapsLayer;
import org.neuroph.nnet.comp.layer.Layer2D;
import org.neuroph.util.ConnectionFactory;
import org.neuroph.util.NeuronProperties;
import org.neuroph.util.TransferFunctionType;


public class geneNetworkCNNLayer extends FeatureMapsLayer {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -5710965559171230011L;
	
	/**
     * Default neuron properties for convolutional layer
     */
    public static final NeuronProperties DEFAULT_NEURON_PROP = new NeuronProperties();
    protected ArrayList<ArrayList<Integer>> geneNetwork;

    static {
        DEFAULT_NEURON_PROP.setProperty("useBias", true);
//      DEFAULT_NEURON_PROP.setProperty("transferFunction", Tanh.class);
        DEFAULT_NEURON_PROP.setProperty("transferFunction", RectifiedLinear.class);
//        DEFAULT_NEURON_PROP.setProperty("transferFunction", TransferFunctionType.SIGMOID);
        DEFAULT_NEURON_PROP.setProperty("inputFunction", WeightedSum.class);
    }

	public geneNetworkCNNLayer(FeatureMapsLayer fromLayer, Kernel kernel,
			ArrayList<ArrayList<Integer>> geneNetwork) {
		super(kernel);
		this.geneNetwork = geneNetwork;

        Layer2D.Dimensions fromDimension = fromLayer.getMapDimensions();
        int mapWidth = fromDimension.getWidth();
        int mapHeight = 1;
        this.mapDimensions = new Layer2D.Dimensions(mapWidth, mapHeight);

        createFeatureMaps(1, this.mapDimensions, ConvolutionalLayer.DEFAULT_NEURON_PROP);
	}

	public geneNetworkCNNLayer(FeatureMapsLayer fromLayer, Kernel kernel,
			int numberOfMaps,
			ArrayList<ArrayList<Integer>> geneNetwork, int hiddenLayerSize) {
		super(kernel);
		this.geneNetwork = geneNetwork;

//        Layer2D.Dimensions fromDimension = fromLayer.getMapDimensions();
        int mapWidth = hiddenLayerSize;
        int mapHeight = 1;
        this.mapDimensions = new Layer2D.Dimensions(mapWidth, mapHeight);

        createFeatureMaps(numberOfMaps, this.mapDimensions, ConvolutionalLayer.DEFAULT_NEURON_PROP);
	}

	public geneNetworkCNNLayer(FeatureMapsLayer fromLayer, Kernel kernel,
			int numberOfMaps, NeuronProperties neuronProp,
			ArrayList<ArrayList<Integer>> geneNetwork) {
		super(kernel);
		this.geneNetwork = geneNetwork;

        Layer2D.Dimensions fromDimension = fromLayer.getMapDimensions();
        int mapWidth = fromDimension.getWidth();
        int mapHeight = 1;
        this.mapDimensions = new Layer2D.Dimensions(mapWidth, mapHeight);

        createFeatureMaps(numberOfMaps, this.mapDimensions,neuronProp);
	}
	
	@Override
    public void connectMaps(Layer2D fromMap, Layer2D toMap) {

//        Weight[][] weights = new Weight[kernel.getHeight()][kernel.getWidth()];
//
//        for (int i = 0; i < kernel.getHeight(); i++) {
//            for (int j = 0; j < kernel.getWidth(); j++) {
//                Weight weight = new Weight();
//                weight.randomize(-0.10, 0.10);
//                weights[i][j] = weight;
//            }
//        }
//        kernel.setWeights(weights);
//        for (int x = 0; x < toMap.getWidth(); x++) { // iterate all neurons by width in toMap
//            Neuron toNeuron = toMap.getNeuronAt(x, 0); // get neuron at specified position in toMap
//        	for(int kx =0; kx < geneNetwork.get(x).size(); kx++){
//        		if(kx>=kernel.getWidth()) break;
//                int fromX = geneNetwork.get(x).get(kx);
//                Weight[][] concreteKernel = kernel.getWeights();
//                Neuron fromNeuron = fromMap.getNeuronAt(fromX, 0);
//                ConnectionFactory.createConnection(fromNeuron, toNeuron, concreteKernel[0][kx]);
//        	}
//        }
		Weight weight;
        for(int x = 0;x < fromMap.getWidth();x++) {
        	Neuron fromNeuron = fromMap.getNeuronAt(x, 0);
        	for(int kx =0; kx < geneNetwork.get(x).size(); kx++){
        		if(kx>=kernel.getWidth()) break;
        		Neuron toNeuron = toMap.getNeuronAt(geneNetwork.get(x).get(kx), 0);
        		weight = new Weight();
        		weight.randomize(0.2d,0.5d);
        		ConnectionFactory.createConnection(fromNeuron, toNeuron, weight);
        	}
        }
    }

    private double getWeightCoeficient(Layer2D toMap) {
        int numberOfInputConnections = toMap.getNeuronAt(0, 0).getInputConnections().length;
        double coefficient = 1d / Math.sqrt(numberOfInputConnections);
        coefficient = !Double.isInfinite(coefficient) || !Double.isNaN(coefficient) || coefficient == 0 ? 1 : coefficient;
        return coefficient;
    }
}
