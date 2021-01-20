package cupdnn.layer;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import cupdnn.Network;
import cupdnn.data.Blob;
/*
 * InputLayer主要作用是占据第一个位置，是的反向传播的算法更容易实现
 */
import cupdnn.data.BlobParams;

/**
 * 输入层
 */
public class InputLayer extends Layer {
    public static final String TYPE = "InputLayer";

    Network mNetwork;
    int width;
    int height;
    /**
     * tensorflow : 对于输入样本中 channels 的含义。一般的RGB图片，channels 数量是 3 （红、绿、蓝）；而monochrome图片，channels 数量是 1 。
     * mxnet : channels 的含义是，每个卷积层中卷积核的数量。
     */
    int channel;

    public InputLayer(Network network) {
        super(network);
        this.mNetwork = network;
    }

    public InputLayer(Network network, int width, int height, int channel) {
        super(network);
        this.width = width;
        this.height = height;
        this.channel = channel;
        this.mNetwork = network;
    }

    @Override
    public String getType() {
        return TYPE;
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

    public void setInputData(Blob input) {
        Blob curData = mNetwork.getDatas().get(id);
        input.cloneTo(curData);
    }

    @Override
    public void saveModel(ObjectOutputStream out) {
        try {
            out.writeUTF(getType());
            //保存的时候，batch也就是layerParams的number总是1，因为predict的时候，因为真正使用的时候，这个batch一般都是1
            out.writeInt(width);
            out.writeInt(height);
            out.writeInt(channel);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void loadModel(ObjectInputStream in) {
        try {
            width = in.readInt();
            height = in.readInt();
            channel = in.readInt();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public Blob createOutBlob() {
        return new Blob(mNetwork.getBatch(), channel, height, width);
    }

    @Override
    public Blob createDiffBlob() {
        return new Blob(mNetwork.getBatch(), channel, height, width);
    }

    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }

    public int getChannel() {
        return channel;
    }
}
