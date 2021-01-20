package cupdnn.optimizer;


import cupdnn.data.Blob;

/**
 * 梯度优化器-SGD
 * 特点：没有使用动量，收敛慢，容易陷入局部极值。
 */
public class SGDOptimizer extends Optimizer {

    public SGDOptimizer(float lr) {
        super(lr);
    }


    public SGDOptimizer(float lr, Optimizer.GMode mode, float lamda) {
        super(lr, mode, lamda);
    }

    @Override
    public void updateB(Blob b, Blob gradient) {
        float[] bData = b.getData();
        float[] gradData = gradient.getData();
        for (int j = 0; j < b.getSize(); j++) {
            bData[j] -= learnRate * gradData[j];
        }
    }

    @Override
    public void updateW(Blob w, Blob gradient) {
        float[] wData = w.getData();
        float[] gradData = gradient.getData();
        switch (mode) {
            case L1:{
                for (int j = 0; j < w.getSize(); j++) {
                    //添加l1衰减
                    if (wData[j] >= 0) {
                        wData[j] = wData[j] - learnRate * lamda - learnRate * gradData[j];
                    } else {
                        wData[j] = wData[j] + learnRate * lamda - learnRate * gradData[j];
                    }
                }
            }break;
            case L2:{
                for (int j = 0; j < w.getSize(); j++) {
                    //添加l2衰减
                    wData[j] = (1.0f - learnRate * lamda) * wData[j] - learnRate * gradData[j];
                }
            }break;
            case NONE:
            default:{
                for (int j = 0; j < w.getSize(); j++) {
                    wData[j] -= learnRate * gradData[j];
                }
            }break;
        }
    }
}
