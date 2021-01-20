package cupdnn.layer;

import cupdnn.Network;
import cupdnn.data.Blob;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.List;

/**
 * 最大值池化层
 * 卷积层参数误差造成估计均值的偏移，max能减小这种误差。
 * max对纹理提取更好如果是识别字体什么的，应该考虑max.
 * 池化层(Max Pooling Layer) - 主要作用是下采样(downsampling)，却不会损坏识别结果。
 */
public class PoolMaxLayer extends Layer {
    public static final String TYPE = "PoolMaxLayer";

    private Blob maxIndex;
    private int width;
    private int height;
    private int inChannel;
    private int kernelSize;
    /**
     * 步幅
     * Stride的作用：是成倍缩小尺寸，而这个参数的值就是缩小的具体倍数，比如步幅为2，输出就是输入的1/2；步幅为3，输出就是输入的1/3
     * https://blog.csdn.net/weixin_42899627/article/details/108228008
     */
    private int stride;

    public PoolMaxLayer(Network network) {
        super(network);
    }

    public PoolMaxLayer(Network network, int width, int height, int inChannel, int kernelSize, int stride) {
        super(network);
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
        maxIndex = new Blob(mNetwork.getBatch(), inChannel, height, width);

    }

    @Override
    public void forward() {
        Blob input = mNetwork.getDatas().get(id - 1);
        Blob output = mNetwork.getDatas().get(id);
        float[] outputData = output.getData();
        float[] inputData = input.getData();
        float[] maxIndexData = maxIndex.getData();
        List<Runnable> tasks = new ArrayList<>();
        for (int n = 0; n < output.getNumbers(); n++) {
            int finalN = n;
            tasks.add(() -> {
                for (int c = 0; c < output.getChannels(); c++) {
                    for (int h = 0; h < output.getHeight(); h++) {
                        for (int w = 0; w < output.getWidth(); w++) {
                            int inStartX = w * stride;
                            int inStartY = h * stride;
                            float localMaxVlue = 0;
                            int localMaxIndex = 0;
                            for (int kh = 0; kh < kernelSize; kh++) {
                                for (int kw = 0; kw < kernelSize; kw++) {
                                    int curIndex = input.getIndexByParams(finalN, c, inStartY + kh, inStartX + kw);
                                    if (inputData[curIndex] > localMaxVlue) {
                                        localMaxVlue = inputData[curIndex];
                                        localMaxIndex = kh * kernelSize + kw;
                                    }
                                }
                            }
                            maxIndexData[maxIndex.getIndexByParams(finalN, c, h, w)] = localMaxIndex;
                            outputData[output.getIndexByParams(finalN, c, h, w)] = localMaxVlue;
                        }
                    }
                }
            });
        }
        mNetwork.runTasksAndWait(tasks);
    }

    @Override
    public void backward() {
        Blob inputDiff = mNetwork.getDiffs().get(id);
        Blob outputDiff = mNetwork.getDiffs().get(id - 1);
        float[] inputDiffData = inputDiff.getData();
        float[] outputDiffData = outputDiff.getData();
        float[] maxIndexData = maxIndex.getData();

        List<Runnable> workers = new ArrayList<>();
        for (int n = 0; n < inputDiff.getNumbers(); n++) {
            int finalN = n;
            workers.add(() -> {
                for (int c = 0; c < inputDiff.getChannels(); c++) {
                    for (int h = 0; h < inputDiff.getHeight(); h++) {
                        for (int w = 0; w < inputDiff.getWidth(); w++) {
                            int inStartX = w * stride;
                            int inStartY = h * kernelSize;
                            int iY = (int) maxIndexData[maxIndex.getIndexByParams(finalN, c, h, w)] / kernelSize;
                            int iX = (int) maxIndexData[maxIndex.getIndexByParams(finalN, c, h, w)] % kernelSize;

                            outputDiffData[outputDiff.getIndexByParams(finalN, c, inStartY + iY, inStartX + iX)] = inputDiffData[inputDiff.getIndexByParams(finalN, c, h, w)];
                        }
                    }
                }
            });
        }
        mNetwork.runTasksAndWait(workers);
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
