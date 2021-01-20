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
 * 分类器
 * Softmax Layer用来将实数输出normalize成多类的概率
 */
public class SoftMaxLayer extends Layer {
    public static final String TYPE = "SoftMaxLayer";

    private int size;

    public SoftMaxLayer(Network network) {
        super(network);
    }

    public SoftMaxLayer(Network network, int size) {
        super(network);
        this.size = size;
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
        float[] inputData = input.getData();
        float[] outputData = output.getData();
        assert input.getSize() == output.getSize() : "SoftMax forward---- input.getSize()==output.getSize() error";
        List<Runnable> tasks = new ArrayList<>();;
        int batch = mNetwork.getBatch();
        for (int n = 0; n < batch; n++) {
            int finalN = n;
            tasks.add(()->{
                float sum = 0.0f;
                float max = 0.001f;

                //查找最大值
                for (int is = 0; is < input.getWidth(); is++) {
                    max = Math.max(max, inputData[finalN * input.getWidth() + is]);
                }
                //求和
                for (int is = 0; is < input.getWidth(); is++) {
                    outputData[finalN * input.getWidth() + is] = (float) Math.exp(inputData[finalN * input.getWidth() + is] - max);
                    sum += outputData[finalN * input.getWidth() + is];
                }
                if (sum == 0) {
                    System.out.println("sum is zero");
                    System.exit(0);
                }
                //每一项除以sum
                for (int os = 0; os < output.getWidth(); os++) {
                    outputData[finalN * output.getWidth() + os] /= sum;
                }

//					//求和
//					for(int is=0;is<input.get3DSize();is++){
//						sum += Math.exp(inputData[n*input.get3DSize()+is]);
//					}
//					//每一项除以sum
//					for(int os=0;os<output.get3DSize();os++){
//						outputData[n*output.get3DSize()+os] = Math.exp(inputData[n*output.get3DSize()+os])/sum;
//					}
            });
        }
        mNetwork.runTasksAndWait(tasks);
    }

    @Override
    public void backward() {
        Blob inputDiff = mNetwork.getDiffs().get(id);
        Blob outputDiff = mNetwork.getDiffs().get(id - 1);
        Blob output = mNetwork.getDatas().get(id);
        float[] inputDiffData = inputDiff.getData();
        float[] outputDiffData = outputDiff.getData();
        float[] outputData = output.getData();
        assert inputDiff.getSize() == outputDiff.getSize() : "SoftMax backward---- inputDiff.getSize()==outputDiff.getSize() error";

        //先求softmax函数的偏导数
        outputDiff.fillValue(0);
        List<Runnable> tasks = new ArrayList<>();
        int batch = mNetwork.getBatch();
        for (int n = 0; n < batch; n++) {
            int finalN = n;
            tasks.add(()->{
                for (int ods = 0; ods < outputDiff.getWidth(); ods++) {
                    for (int ids = 0; ids < inputDiff.getWidth(); ids++) {
                        if (ids == ods) {
                            outputDiffData[finalN * output.getWidth() + ods] += outputData[finalN * output.getWidth() + ods] * (1.0 - outputData[finalN * output.getWidth() + ods])
                                    * inputDiffData[finalN * output.getWidth() + ids];
                        } else {
                            outputDiffData[finalN * output.getWidth() + ods] -= outputData[finalN * output.getWidth() + ods] * outputData[finalN * output.getWidth() + ids]
                                    * inputDiffData[finalN * output.getWidth() + ids];
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
            out.writeInt(size);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void loadModel(ObjectInputStream in) {
        try {
            size = in.readInt();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public Blob createOutBlob() {
        return new Blob(mNetwork.getBatch(), size);
    }

    @Override
    public Blob createDiffBlob() {
        return new Blob(mNetwork.getBatch(), size);
    }
}
