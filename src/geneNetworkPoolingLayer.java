import org.neuroph.core.Neuron;
import org.neuroph.core.Weight;
import org.neuroph.core.input.Max;
import org.neuroph.core.transfer.Tanh;
import org.neuroph.nnet.comp.Kernel;
import org.neuroph.nnet.comp.layer.FeatureMapsLayer;
import org.neuroph.nnet.comp.layer.Layer2D;
import org.neuroph.util.ConnectionFactory;
import org.neuroph.util.NeuronProperties;


public class geneNetworkPoolingLayer extends FeatureMapsLayer {

    public static final NeuronProperties DEFAULT_NEURON_PROP = new NeuronProperties();

    static {
        DEFAULT_NEURON_PROP.setProperty("useBias", true);
        DEFAULT_NEURON_PROP.setProperty("transferFunction", Tanh.class);
        DEFAULT_NEURON_PROP.setProperty("inputFunction", Max.class);
    }

    /**
     * Creates pooling layer with specified kernel, appropriate map
     * dimensions in regard to previous layer (fromLayer param) and specified
     * number of feature maps with default neuron settings for pooling layer.
     * Number of maps in pooling layer must be the same as number of maps in previous
     * layer.
     *
     * @param fromLayer previous layer, which will be connected to this layer
     * @param kernel    kernel for all feature maps
     */
    public geneNetworkPoolingLayer(FeatureMapsLayer fromLayer, Kernel kernel) {
        super(kernel);
        int numberOfMaps = fromLayer.getNumberOfMaps();
        Layer2D.Dimensions fromDimension = fromLayer.getMapDimensions();

        int mapWidth = fromDimension.getWidth() / kernel.getWidth();
        int mapHeight = fromDimension.getHeight() / kernel.getHeight();
        this.mapDimensions = new Layer2D.Dimensions(mapWidth, mapHeight);

        createFeatureMaps(numberOfMaps, mapDimensions, DEFAULT_NEURON_PROP);
    }

    /**
     * Creates pooling layer with specified kernel, appropriate map
     * dimensions in regard to previous layer (fromLayer param) and specified
     * number of feature maps with given neuron properties.
     *
     * @param fromLayer    previous layer, which will be connected to this layer
     * @param kernel       kernel for all feature maps
     * @param numberOfMaps number of feature maps to create in this layer
     * @param neuronProp   settings for neurons in feature maps
     */
    public geneNetworkPoolingLayer(FeatureMapsLayer fromLayer, Kernel kernel, int numberOfMaps, NeuronProperties neuronProp) {
        super(kernel);
        Layer2D.Dimensions fromDimension = fromLayer.getMapDimensions();

        int mapWidth = fromDimension.getWidth() / kernel.getWidth();
        int mapHeight = fromDimension.getHeight() / kernel.getHeight();
        this.mapDimensions = new Layer2D.Dimensions(mapWidth, mapHeight);

        createFeatureMaps(numberOfMaps, mapDimensions, neuronProp);
    }

    /**
     * Creates connections with shared weights between two feature maps
     * Assumes that toMap is from Pooling layer.
     * <p/>
     * In this implementation, there is no overlapping between kernel positions.
     *
     * @param fromMap source feature map
     * @param toMap   destination feature map
     */
    @Override
    public void connectMaps(Layer2D fromMap, Layer2D toMap) {
        int kernelWidth = kernel.getWidth();
        int kernelHeight = kernel.getHeight();
        Weight weight = new Weight();
        weight.setValue(1);
        for (int x = 0; x < fromMap.getWidth() - kernelWidth + 1; x += kernelWidth) {
            for (int y = 0; y < fromMap.getHeight() - kernelHeight + 1; y += kernelHeight) {

                Neuron toNeuron = toMap.getNeuronAt(x / kernelWidth, y / kernelHeight);
                for (int dy = 0; dy < kernelHeight; dy++) {
                    for (int dx = 0; dx < kernelWidth; dx++) {
                        int fromX = x + dx;
                        int fromY = y + dy;
                        Neuron fromNeuron = fromMap.getNeuronAt(fromX, fromY);
                        ConnectionFactory.createConnection(fromNeuron, toNeuron, weight);
                    }
                }
            }
        }
    }
}
