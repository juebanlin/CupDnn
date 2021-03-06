package cupdnn.optimizer;

import java.util.HashMap;

import cupdnn.data.Blob;

/**
 * 优化器-SGD with Momentum带动量的SGD
 * 特点：利用了梯度的动量，收敛比SGD快
 */
public class SGDMOptimizer extends Optimizer {

    /**
     * 大或等于0的浮点数，学习率
     */
    private float momentum = 0.9f;
    private HashMap<Blob, Blob> privMap = new HashMap();

    public SGDMOptimizer(float learnRate, float mententum) {
        super(learnRate);
        this.momentum = mententum;
    }

    /*
     * lamda是衰减权重，是一个很小的数字
     * */

    public SGDMOptimizer(float learnRate, Optimizer.GMode mode, float lamda, float mententum) {
        super(learnRate, mode, lamda);
        this.momentum = mententum;
    }

    @Override
    public void updateB(Blob b, Blob gradient) {
        Blob priv = privMap.get(b);
        if (priv == null) {
            priv = new Blob(b, false);
            privMap.put(b, priv);
        }
        float[] privData = priv.getData();
        float[] bData = b.getData();
        float[] gradData = gradient.getData();
        for (int j = 0; j < b.getSize(); j++) {
            float V = momentum * privData[j] - learnRate * gradData[j];
            bData[j] += V;
            privData[j] = V;
        }
    }

    @Override
    public void updateW(Blob w, Blob gradient) {
        Blob priv = privMap.get(w);
        if (priv == null) {
            priv = new Blob(w, false);
            privMap.put(w, priv);
        }
        float[] privData = priv.getData();
        float[] wData = w.getData();
        float[] gradData = gradient.getData();
        switch (mode) {
            case L1: {
                for (int j = 0; j < w.getSize(); j++) {
                    //添加l1衰减
                    float V = 0;
                    if (wData[j] >= 0) {
                        V = momentum * privData[j] - learnRate * lamda - learnRate * gradData[j];
                    } else {
                        V = momentum * privData[j] + learnRate * lamda - learnRate * gradData[j];
                    }
                    wData[j] += V;
                    privData[j] = V;
                }
            }
            break;
            case L2: {
                for (int j = 0; j < w.getSize(); j++) {
                    //添加l2衰减
                    float V = momentum * privData[j] - learnRate * lamda * wData[j] - learnRate * gradData[j];
                    wData[j] += V;
                    privData[j] = V;
                }
            }
            case NONE:
            default: {
                for (int j = 0; j < w.getSize(); j++) {
                    float V = momentum * privData[j] - learnRate * gradData[j];
                    wData[j] += V;
                    privData[j] = V;
                }
            }
            break;
        }
    }
}
