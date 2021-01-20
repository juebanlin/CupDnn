package cupdnn.optimizer;

import cupdnn.data.Blob;

/**
 * 梯度优化器
 */
public abstract class Optimizer {
    /**
     * 大或等于0的浮点数，学习率
     */
    protected float learnRate = 0.0f;
    /**
     * 学习率步长
     */
    protected float lamda = 0.0f;

    public enum GMode {
        NONE,
        L1,
        L2
    }

    GMode mode;

    public Optimizer(float lr) {
        this.learnRate = lr;
        this.mode = GMode.NONE;
    }


    public Optimizer(float lr, GMode mode, float lamda) {
        this.learnRate = lr;
        this.lamda = lamda;
        this.mode = mode;
    }

    public abstract void updateW(Blob params, Blob gradient);

    public abstract void updateB(Blob params, Blob gradient);

    public void setLearnRate(float learnRate) {
        this.learnRate = learnRate;
    }

    public float getLearnRate() {
        return this.learnRate;
    }
}
