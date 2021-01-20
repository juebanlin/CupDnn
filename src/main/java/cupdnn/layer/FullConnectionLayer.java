package cupdnn.layer;

import cupdnn.data.Blob;
import cupdnn.data.BlobParams;
import cupdnn.util.MathFunctions;
import cupdnn.util.Task;
import cupdnn.util.ThreadPoolManager;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

import cupdnn.Network;
import cupdnn.active.ReluActivationFunc;
import cupdnn.active.SigmodActivationFunc;
import cupdnn.active.TanhActivationFunc;

/**
 * 全连接层(Fully Connected Layer) - 主要作用是分类。
 */
public class FullConnectionLayer extends Layer {
    public static final String TYPE = "FullConnectionLayer";
    private Blob w;
    private transient Blob wGradient;
    private Blob b;
    private transient Blob bGradient;
    private transient Blob z;
    private int inSize;
    private int outSize;

    public FullConnectionLayer(Network network) {
        super(network);
    }

    public FullConnectionLayer(Network network, int inSize, int outSize) {
        super(network);
        this.inSize = inSize;
        this.outSize = outSize;
    }

    @Override
    public void prepare() {
        if (w == null && b == null) {
            //表明该层公有outSize个神经元，每个神经元和前面层的inSize个神经元向连
            w = new Blob(inSize, outSize);

            //表明该层有outSize个神经元，每个神经元有一个偏执
            b = new Blob(outSize);


            //初始化
            float[] wData = w.getData();
            float[] bData = b.getData();
            //高斯分布初始化w
            MathFunctions.gaussianInitData(wData);
            //常量初始化b
            MathFunctions.constantInitData(bData, 0.001f);
        }
        wGradient = new Blob(inSize, outSize);
        bGradient = new Blob(outSize);
        //z是个中间值，计算的时候要用到。
        z = new Blob(mNetwork.getBatch(), outSize);
    }

    @Override
    public void forward() {
        Blob input = mNetwork.getDatas().get(id - 1);
        Blob output = mNetwork.getDatas().get(id);
        float[] inputData = input.getData();
        float[] outputData = output.getData();
        float[] wData = w.getData();
        float[] bData = b.getData();
        float[] zData = z.getData();
        z.fillValue(0);
        List<Runnable> tasks = new ArrayList<>();
        int batch = mNetwork.getBatch();
        for (int n = 0; n < batch; n++) {
            int finalN = n;
            tasks.add(()->{
                for (int os = 0; os < outSize; os++) {//有多少个输出，当前层就有多少个神经元
                    //和每个神经元的权重相乘
                    for (int is = 0; is < inSize; is++) {
                        //zData[n*output.get3DSize()+os] 表示一个批次中的第n个的第os个神经元
                        zData[finalN * outSize + os] += inputData[finalN * inSize + is] * wData[os * inSize + is];
                    }
                    //偏执
                    zData[finalN * outSize + os] += bData[os];
                    //激活函数
                    if (activationFunc != null) {
                        outputData[finalN * outSize + os] = activationFunc.active(zData[finalN * outSize + os]);
                    } else {
                        outputData[finalN * outSize + os] = zData[finalN * outSize + os];
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
        Blob input = mNetwork.getDatas().get(id - 1);
        float[] inputData = input.getData();
        float[] inputDiffData = inputDiff.getData();
        float[] outputDiffData = outputDiff.getData();
        float[] wData = w.getData();
        float[] wGradientData = wGradient.getData();
        float[] bGradientData = bGradient.getData();
        float[] zData = z.getData();

        //update diff
        //先乘激活函数的偏导数,即可求出当前层的误差
        assert inputDiff.getSize() == z.getSize() : "inputDiff.getSize()==z.getSize() error";
        int batch = mNetwork.getBatch();
        if (activationFunc != null) {
            List<Runnable> tasks = new ArrayList<>();
            for (int n = 0; n < batch; n++) {
                int finalN = n;
                tasks.add(()->{
                    for (int ids = 0; ids < outSize; ids++) {
                        inputDiffData[finalN * outSize + ids] *= activationFunc.diffActive(zData[finalN * outSize + ids]);
                    }
                });
            }
            mNetwork.runTasksAndWait(tasks);
        }
        wGradient.fillValue(0);
        List<Runnable> tasks = new ArrayList<>();
        for (int n = 0; n < batch; n++) {
            int finalN = n;
            tasks.add(()->{
                for (int ids = 0; ids < outSize; ids++) {
                    for (int is = 0; is < inSize; is++) {
                        //相当于一个神经元和它的每一个连接乘加
                        wGradientData[ids * inSize + is] += inputData[finalN * inSize + is] * inputDiffData[finalN * outSize + ids];
                    }
                }
            });
        }
        mNetwork.runTasksAndWait(tasks);
        //平均
        MathFunctions.dataDivConstant(wGradientData, batch);

        //update bias
        bGradient.fillValue(0);
        for (int n = 0; n < batch; n++) {
            for (int bs = 0; bs < outSize; bs++) {
                bGradientData[bs] += inputDiffData[n * outSize + bs];
            }
        }

        //平均
        MathFunctions.dataDivConstant(bGradientData, batch);

        //最后，乘以当前层的权重后输出
        //每一个输出=每一个神经元与连接他的权重的乘加
        if (id <= 1) return;
        outputDiff.fillValue(0);
        tasks.clear();
        for (int n = 0; n < batch; n++) {
            int finalN = n;
            tasks.add(()->{
                for (int ids = 0; ids < outSize; ids++) {
                    for (int ods = 0; ods < inSize; ods++) {
                        outputDiffData[finalN * inSize + ods] += inputDiffData[finalN * outSize + ids] * wData[ids * inSize + ods];
                    }
                }
            });
        }
        mNetwork.runTasksAndWait(tasks);
        mNetwork.updateW(w, wGradient);
        mNetwork.updateW(b, bGradient);
    }


    @Override
    public String getType() {
        return TYPE;
    }

    @Override
    public void saveModel(ObjectOutputStream out) {
        try {
            out.writeUTF(getType());
            out.writeInt(inSize);
            out.writeInt(outSize);
            out.writeObject(w);
            out.writeObject(b);
            if (activationFunc != null) {
                out.writeUTF(activationFunc.getType());
            } else {
                out.writeUTF("none");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    @Override
    public void loadModel(ObjectInputStream in) {
        try {
            inSize = in.readInt();
            outSize = in.readInt();
            w = (Blob) in.readObject();
            b = (Blob) in.readObject();
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
        return new Blob(mNetwork.getBatch(), outSize);
    }

    @Override
    public Blob createDiffBlob() {
        return new Blob(mNetwork.getBatch(), outSize);
    }
}
