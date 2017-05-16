import org.neuroph.nnet.comp.layer.FeatureMapsLayer;
import org.neuroph.nnet.comp.layer.Layer2D;
import org.neuroph.nnet.comp.layer.PoolingLayer;


public class geneNetworkCNNUtils {

	public static void fullConnectMapLayers(FeatureMapsLayer fromLayer, FeatureMapsLayer toLayer) {
        if (toLayer instanceof geneNetworkCNNLayer) {
            for (int i = 0; i < fromLayer.getNumberOfMaps(); i++) {
                for (int j = 0; j < toLayer.getNumberOfMaps(); j++) {
                    Layer2D fromMap = fromLayer.getFeatureMap(i);
                    Layer2D toMap = toLayer.getFeatureMap(j);
                    toLayer.connectMaps(fromMap, toMap);
                }
            }
        } else if (toLayer instanceof geneNetworkPoolingLayer) { 
            for (int i = 0; i < toLayer.getNumberOfMaps(); i++) {
                Layer2D fromMap = fromLayer.getFeatureMap(i);
                Layer2D toMap = toLayer.getFeatureMap(i);
                toLayer.connectMaps(fromMap, toMap);
            }
        }
    }


    /**
     * Creates connections between two feature maps
     *
     * @param fromLayer           parent layer for from feature map
     * @param toLayer             parent layer for to feature map
     * @param fromFeatureMapIndex index of from feature map
     * @param toFeatureMapIndex   index of to feature map
     */
    public static void connectFeatureMaps(FeatureMapsLayer fromLayer, FeatureMapsLayer toLayer,
                                          int fromFeatureMapIndex, int toFeatureMapIndex) {
        Layer2D fromMap = fromLayer.getFeatureMap(fromFeatureMapIndex);
        Layer2D toMap = toLayer.getFeatureMap(toFeatureMapIndex);
        toLayer.connectMaps(fromMap, toMap);
    }

}
