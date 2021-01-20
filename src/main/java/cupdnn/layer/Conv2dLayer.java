package cupdnn.layer;


import cupdnn.Network;
import cupdnn.active.ReluActivationFunc;
import cupdnn.active.SigmodActivationFunc;
import cupdnn.active.TanhActivationFunc;
import cupdnn.data.Blob;
import cupdnn.util.MathFunctions;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.List;

/**
 * 标准卷积
 * 卷积层(Convolutional Layer) - 主要作用是提取特征。
 */
public class Conv2dLayer extends Layer {
    public static final String TYPE = "Conv2dLayer";
    private Blob kernel;
    /**
     * 偏置值
     */
    private Blob bias;
    private Blob kernelGradient;
    private Blob biasGradient;
    private Blob z;
    private int width;
    private int height;
    private int inChannel;
    private int outChannel;
    private int kernelSize;
    /**
     * 步幅
     * Stride的作用：是成倍缩小尺寸，而这个参数的值就是缩小的具体倍数，比如步幅为2，输出就是输入的1/2；步幅为3，输出就是输入的1/3
     * 问题：一个尺寸 a*a 的特征图，经过 b*b 的卷积层，步幅（stride）=c，填充（padding）=d，
     * 请计算出输出的特征图尺寸？
     * 答：若d等于0，也就是不填充，输出的特征图的尺寸=（a-b）/c+1
     * 若d不等于0，也就是填充，输出的特征图的尺寸=（a+2d-b）/c+1
     * 栗子1：  一个特征图尺寸为4*4的输入，使用3*3的卷积核，步幅=1，填充=0
     * 输出的尺寸=(4 - 3)/1 + 1 = 2
     * 栗子2：  一个特征图尺寸为5*5的输入，使用3*3的卷积核，步幅=1，填充=1
     * 输出的尺寸=(5 + 2*1 - 3)/1 + 1 = 5
     * https://blog.csdn.net/weixin_42899627/article/details/108228008
     */
    private int stride;

    public Conv2dLayer(Network network) {
        super(network);
    }

    public Conv2dLayer(Network network, int width, int height, int inChannel, int outChannel, int kernelSize, int stride) {
        super(network);
        this.width = width;
        this.height = height;
        this.inChannel = inChannel;
        this.outChannel = outChannel;
        this.kernelSize = kernelSize;
        this.stride = stride;
    }

    @Override
    public String getType() {
        return TYPE;
    }

    @Override
    public void prepare() {
        //layerParams.getHeight()表示该层需要提取的特征数量
        if (kernel == null && bias == null) {
            kernel = new Blob(inChannel * outChannel, kernelSize, kernelSize);
            bias = new Blob(outChannel);
            //init params
            MathFunctions.gaussianInitData(kernel.getData());
            MathFunctions.constantInitData(bias.getData(), 0.001f);
        }
        z = new Blob(mNetwork.getBatch(), outChannel, height, width);
        kernelGradient = new Blob(inChannel * outChannel, kernelSize, kernelSize);
        biasGradient = new Blob(outChannel);

    }

    @Override
    public void forward() {
        Blob input = mNetwork.getDatas().get(id - 1);
        Blob output = mNetwork.getDatas().get(id);
        float[] outputData = output.getData();
        float[] zData = z.getData();
        if (activationFunc == null) {
            //卷积后的结果存贮在output中
            output.fillValue(0);
            MathFunctions.conv2dBlobSame(mNetwork, input, kernel, bias, output);
            return;
        }
        //激活函数
        //卷积后的结果存贮在z中
        z.fillValue(0);
        MathFunctions.conv2dBlobSame(mNetwork, input, kernel, bias, z);
        List<Runnable> tasks = new ArrayList<>();
        for (int n = 0; n < output.getNumbers(); n++) {
            int finalN = n;
            tasks.add(() -> {
                for (int c = 0; c < output.getChannels(); c++) {
                    for (int h = 0; h < output.getHeight(); h++) {
                        for (int w = 0; w < output.getWidth(); w++) {
                            outputData[output.getIndexByParams(finalN, c, h, w)] = activationFunc.active(zData[z.getIndexByParams(finalN, c, h, w)]);
                        }
                    }
                }
            });
        }
        mNetwork.runTasksAndWait(tasks);
    }

    @Override
    public void backward() {
        Blob input = mNetwork.getDatas().get(id - 1);
        Blob inputDiff = mNetwork.getDiffs().get(id);
        Blob outputDiff = mNetwork.getDiffs().get(id - 1);
        float[] inputDiffData = inputDiff.getData();
        float[] zData = z.getData();
        float[] kernelGradientData = kernelGradient.getData();
        float[] inputData = input.getData();
        float[] biasGradientData = biasGradient.getData();
        //先乘激活函数的导数,得到该层的误差

        if (activationFunc != null) {
            List<Runnable> tasks = new ArrayList<>();
            for (int n = 0; n < inputDiff.getNumbers(); n++) {
                int finalN = n;
                tasks.add(() -> {
                    for (int c = 0; c < inputDiff.getChannels(); c++) {
                        for (int h = 0; h < inputDiff.getHeight(); h++) {
                            for (int w = 0; w < inputDiff.getWidth(); w++) {
                                inputDiffData[inputDiff.getIndexByParams(finalN, c, h, w)] *= activationFunc.diffActive(zData[z.getIndexByParams(finalN, c, h, w)]);
                            }
                        }
                    }
                });
            }
            mNetwork.runTasksAndWait(tasks);
        }

        //然后更新参数
        //计算kernelGradient,这里并不更新kernel,kernel在优化器中更新
        kernelGradient.fillValue(0);
        List<Runnable> tasks = new ArrayList<>();
        for (int n = 0; n < inputDiff.getNumbers(); n++) {
            int finalN = n;
            tasks.add(() -> {
                for (int ci = 0; ci < inputDiff.getChannels(); ci++) {
                    for (int co = 0; co < outputDiff.getChannels(); co++) {
                        for (int h = 0; h < inputDiff.getHeight(); h++) {
                            for (int w = 0; w < inputDiff.getWidth(); w++) {
                                //先定位到输出的位置
                                //然后遍历kernel,通过kernel定位输入的位置
                                //然后将输入乘以diff
                                int inStartX = w - kernelGradient.getWidth() / 2;
                                int inStartY = h - kernelGradient.getHeight() / 2;
                                //和卷积核乘加

                                for (int kh = 0; kh < kernelGradient.getHeight(); kh++) {
                                    for (int kw = 0; kw < kernelGradient.getWidth(); kw++) {
                                        int inY = inStartY + kh;
                                        int inX = inStartX + kw;
                                        if (inY >= 0 && inY < input.getHeight() && inX >= 0 && inX < input.getWidth()) {
                                            kernelGradientData[kernelGradient.getIndexByParams(0, ci * outputDiff.getChannels() + co, kh, kw)] += inputData[input.getIndexByParams(finalN, co, inY, inX)]
                                                    * inputDiffData[inputDiff.getIndexByParams(finalN, ci, h, w)];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            });
        }
        mNetwork.runTasksAndWait(tasks);
        //平均
        MathFunctions.dataDivConstant(kernelGradientData, inputDiff.getNumbers());

        //更新bias
        biasGradient.fillValue(0);
        for (int n = 0; n < inputDiff.getNumbers(); n++) {
            for (int c = 0; c < inputDiff.getChannels(); c++) {
                for (int h = 0; h < inputDiff.getHeight(); h++) {
                    for (int w = 0; w < inputDiff.getWidth(); w++) {
                        biasGradientData[bias.getIndexByParams(0, 0, 0, c)] += inputDiffData[inputDiff.getIndexByParams(n, c, h, w)];
                    }
                }
            }
        }
        //平均
        MathFunctions.dataDivConstant(biasGradientData, inputDiff.getNumbers());

        if (id <= 1) return;
        //先把kernel旋转180度
        //Blob kernelRoate180 = MathFunctions.rotate180Blob(kernel);
        //然后再做卷积
        outputDiff.fillValue(0);
        MathFunctions.conv2dBlobSame(mNetwork, inputDiff, kernel, outputDiff);

        mNetwork.updateW(kernel, kernelGradient);
        mNetwork.updateW(bias, biasGradient);
    }

    @Override
    public void saveModel(ObjectOutputStream out) {
        try {
            out.writeUTF(getType());
            out.writeInt(width);
            out.writeInt(height);
            out.writeInt(inChannel);
            out.writeInt(outChannel);
            out.writeInt(kernelSize);
            out.writeInt(stride);
            out.writeObject(kernel);
            out.writeObject(bias);
            if (activationFunc != null) {
                out.writeUTF(activationFunc.getType());
            }
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
            outChannel = in.readInt();
            kernelSize = in.readInt();
            stride = in.readInt();
            kernel = (Blob) in.readObject();
            bias = (Blob) in.readObject();
            String activationType = in.readUTF();
            if (activationType.equals(ReluActivationFunc.TYPE)) {
                setActivationFunc(new ReluActivationFunc());
            } else if (activationType.equals(SigmodActivationFunc.TYPE)) {
                setActivationFunc(new SigmodActivationFunc());
            } else if (activationType.equals(TanhActivationFunc.TYPE)) {
                setActivationFunc(new TanhActivationFunc());
            }
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public Blob createOutBlob() {
        return new Blob(mNetwork.getBatch(), outChannel, height, width);
    }

    @Override
    public Blob createDiffBlob() {
        return new Blob(mNetwork.getBatch(), outChannel, height, width);
    }
}
