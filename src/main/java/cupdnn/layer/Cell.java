package cupdnn.layer;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import cupdnn.Network;
import cupdnn.data.Blob;

public class Cell extends Layer {

    public Cell(Network network) {
        super(network);
    }

    @Override
    public Blob createOutBlob() {
        return null;
    }

    @Override
    public Blob createDiffBlob() {
        return null;
    }

    @Override
    public String getType() {
        return null;
    }

    @Override
    public void prepare() {
    }

    @Override
    public void forward() {

    }

    @Override
    public void backward() {

    }

    public void forward(Blob in, Blob out) {

    }

    public void backward(Blob in, Blob out, Blob inDiff, Blob outDiff) {

    }

    public void resetState() {

    }

    @Override
    public void saveModel(ObjectOutputStream out) {

    }

    @Override
    public void loadModel(ObjectInputStream in) {

    }
}
