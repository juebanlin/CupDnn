package cupdnn.layer;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import cupdnn.Network;
import cupdnn.data.Blob;

/**
 * 特征提取层
 * 语音识别用Recurrent layer去学习“如何”去听，再用学好的听取方式去处理声音再送入分类器中
 */
public class RecurrentLayer extends Layer {
    Network mNetwork;
    Cell mCell;
    int seqLen;
    int inSize;
    int hidenSize;
    public static final String TYPE = "RecurrentLayer";

    public static enum RecurrentType {
        RNN,
        LSTM,
        GRU
    }

    RecurrentType type;

    public RecurrentLayer(Network network) {
        super(network);
        this.mNetwork = network;
    }

    public RecurrentLayer(Network network, RecurrentType type, int seqLen, int inSize, int hidenSize) {
        this(network);
        this.seqLen = seqLen;
        this.inSize = inSize;
        this.hidenSize = hidenSize;
        this.type = type;
    }

    @Override
    public Blob createOutBlob() {
        return new Blob(seqLen, mNetwork.getBatch() / seqLen, hidenSize);
    }

    @Override
    public Blob createDiffBlob() {
        return new Blob(seqLen, mNetwork.getBatch() / seqLen, hidenSize);
    }

    @Override
    public String getType() {
        return TYPE;
    }

    @Override
    public void prepare() {
        switch (type) {
            case RNN:
                if (mCell == null) {
                    mCell = new RnnCell(mNetwork, this, inSize, hidenSize);
                }
                mCell.prepare();
                break;
            case LSTM:
                break;
            case GRU:
                break;
        }
    }

    @Override
    public void forward() {
        Blob input = mNetwork.getDatas().get(id - 1);
        Blob output = mNetwork.getDatas().get(id);
        float[] inputData = input.getData();
        float[] outputData = output.getData();
        Blob tmpIn = new Blob(mNetwork.getBatch() / seqLen, inSize);
        Blob tmpOut = new Blob(mNetwork.getBatch() / seqLen, hidenSize);
        for (int i = 0; i < seqLen; i++) {
            if (i == 0) {
                mCell.resetState();
            }
            float[] tmpInData = tmpIn.getData();
            int tmpInSize = tmpIn.getSize();
            //每次去取一个序列
            for (int j = 0; j < tmpInSize; j++) {
                tmpInData[j] = inputData[i * tmpInSize + j];
            }
            mCell.forward(tmpIn, tmpOut);
            float[] tmpOutData = tmpOut.getData();
            int tmpOutSize = tmpOut.getSize();
            for (int j = 0; j < tmpOutSize; j++) {
                outputData[i * tmpOutSize + j] = tmpOutData[j];
            }
        }
    }

    @Override
    public void backward() {
        Blob inputDiff = mNetwork.getDiffs().get(id);
        Blob outputDiff = mNetwork.getDiffs().get(id - 1);
        Blob input = mNetwork.getDatas().get(id - 1);
        Blob output = mNetwork.getDatas().get(id);
        Blob tmpIn = new Blob(mNetwork.getBatch() / seqLen, inSize);
        Blob tmpOut = new Blob(mNetwork.getBatch() / seqLen, hidenSize);
        Blob tmpInDiff = new Blob(mNetwork.getBatch() / seqLen, hidenSize);
        Blob tmpOutDiff = new Blob(mNetwork.getBatch() / seqLen, inSize);
        float[] inputData = input.getData();
        float[] outputData = output.getData();
        float[] inputDiffData = inputDiff.getData();
        float[] outputDiffData = outputDiff.getData();
        for (int i = 0; i < seqLen; i++) {
            //一次取序列中的一个
            float[] tmpInData = tmpIn.getData();
            int tmpInSize = tmpIn.getSize();
            for (int j = 0; j < tmpInSize; j++) {
                tmpInData[j] = inputData[i * tmpInSize + j];
            }
            float[] tmpOutData = tmpOut.getData();
            int tmpOutSize = tmpOut.getSize();
            for (int j = 0; j < tmpOutSize; j++) {
                tmpOutData[j] = outputData[i * tmpOutSize + j];
            }
            float[] tmpInDiffData = tmpInDiff.getData();
            int tmpInDiffSize = tmpInDiff.getSize();
            for (int j = 0; j < tmpInDiffSize; j++) {
                tmpInDiffData[j] = inputDiffData[i * tmpInDiffSize + j];
            }
            mCell.backward(tmpIn, tmpOut, tmpInDiff, tmpOutDiff);
            //将计算的结果按顺序拷贝回outputDiffBlob
            float[] tmpOutDiffData = tmpOutDiff.getData();
            int tmpOutDifftSize = tmpOutDiff.getSize();
            for (int j = 0; j < tmpOutDifftSize; j++) {
                outputDiffData[i * tmpOutDifftSize + j] = tmpOutDiffData[j];
            }
        }
    }

    @Override
    public void saveModel(ObjectOutputStream out) {
        try {
            out.writeUTF(getType());
            out.writeInt(seqLen);
            out.writeInt(inSize);
            out.writeInt(hidenSize);
            mCell.saveModel(out);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void loadModel(ObjectInputStream in) {
        try {
            seqLen = in.readInt();
            inSize = in.readInt();
            hidenSize = in.readInt();
            String type = in.readUTF();
            if (type.equals("RNN")) {
                this.type = RecurrentType.RNN;
                mCell = new RnnCell(mNetwork, RecurrentLayer.this);
                mCell.loadModel(in);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public int getSeqLen() {
        return seqLen;
    }
}
