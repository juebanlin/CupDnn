package cupdnn.layer;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

import cupdnn.Network;
import cupdnn.data.Blob;
import cupdnn.data.BlobParams;
import cupdnn.util.Task;
import cupdnn.util.ThreadPoolManager;

/**
 * 平均值池化层
 * 邻域大小受限造成的估计值方差增大，mean能减小这种误差。
 * mean对背景保留更好
 */
public class PoolMeanLayer extends Layer {
    public static final String TYPE = "PoolMeanLayer";
    private Network mNetwork;
    private int width;
    private int height;
    private int inChannel;
    private int kernelSize;
    private int stride;

    public PoolMeanLayer(Network network) {
        super(network);
        mNetwork = network;
    }

    public PoolMeanLayer(Network network, int width, int height, int inChannel, int kernelSize, int stride) {
        super(network);
        this.mNetwork = network;
        this.width = width;
        this.height = height;
        this.inChannel = inChannel;
        this.kernelSize = kernelSize;
        this.stride = stride;
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
        Blob input = mNetwork.getDatas().get(id - 1);
        Blob output = mNetwork.getDatas().get(id);
        float[] outputData = output.getData();
        float[] inputData = input.getData();
        List<Runnable> tasks = new ArrayList<>();
        for (int n = 0; n < output.getNumbers(); n++) {
            int finalN = n;
            tasks.add(()->{for (int c = 0; c < output.getChannels(); c++) {
                for (int h = 0; h < output.getHeight(); h++) {
                    for (int w = 0; w < output.getWidth(); w++) {
                        int inStartX = w * stride;
                        int inStartY = h * stride;
                        float sum = 0;
                        for (int kh = 0; kh < kernelSize; kh++) {
                            for (int kw = 0; kw < kernelSize; kw++) {
                                int curIndex = input.getIndexByParams(finalN, c, inStartY + kh, inStartX + kw);
                                sum += inputData[curIndex];
                            }
                        }
                        outputData[output.getIndexByParams(finalN, c, h, w)] = sum / (kernelSize * kernelSize);
                    }
                }
            }});
        }
        mNetwork.runTasksAndWait(tasks);
    }

    @Override
    public void backward() {
        Blob inputDiff = mNetwork.getDiffs().get(id);
        Blob outputDiff = mNetwork.getDiffs().get(id - 1);
        float[] inputDiffData = inputDiff.getData();
        float[] outputDiffData = outputDiff.getData();
        List<Runnable> tasks = new ArrayList<>();
        for (int n = 0; n < inputDiff.getNumbers(); n++) {
            int finalN = n;
            tasks.add(()->{
                for (int c = 0; c < inputDiff.getChannels(); c++) {
                    for (int h = 0; h < inputDiff.getHeight(); h++) {
                        for (int w = 0; w < inputDiff.getWidth(); w++) {
                            int inStartX = w * stride;
                            int inStartY = h * stride;
                            for (int kh = 0; kh < kernelSize; kh++) {
                                for (int kw = 0; kw < kernelSize; kw++) {
                                    int curIndex = outputDiff.getIndexByParams(finalN, c, inStartY + kh, inStartX + kw);
                                    outputDiffData[curIndex] = inputDiffData[inputDiff.getIndexByParams(finalN, c, h, w)];
                                }
                            }
                        }
                    }
                }
            });
        }
        mNetwork.runTasksAndWait(tasks);
    }

    @Override
    public void saveModel(ObjectOutputStream out) {
        try {
            out.writeUTF(getType());
            out.writeInt(width);
            out.writeInt(height);
            out.writeInt(inChannel);
            out.writeInt(kernelSize);
            out.writeInt(stride);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void loadModel(ObjectInputStream in) {
        try {
            width = in.readInt();
            height = in.readInt();
            inChannel = in.readInt();
            kernelSize = in.readInt();
            stride = in.readInt();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public Blob createOutBlob() {
        return new Blob(mNetwork.getBatch(), inChannel, width / 2, height / 2);
    }

    @Override
    public Blob createDiffBlob() {
        return new Blob(mNetwork.getBatch(), inChannel, width / 2, height / 2);
    }
}
