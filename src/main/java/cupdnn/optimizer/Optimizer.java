package cupdnn.optimizer;

import cupdnn.data.Blob;

/**
 * 梯度优化器
 */
public abstract class Optimizer {
    /**
     * 大或等于0的浮点数，学习率
     */
    protected float lr = 0.0f;
    /**
     * 步长
     */
    protected float lamda = 0.0f;

    public enum GMode {
        NONE,
        L1,
        L2
    }

    GMode mode;

    public Optimizer(float lr) {
        this.lr = lr;
        this.mode = GMode.NONE;
    }


    public Optimizer(float lr, GMode mode, float lamda) {
        this.lr = lr;
        this.lamda = lamda;
        this.mode = mode;
    }

    public abstract void updateW(Blob params, Blob gradient);

    public abstract void updateB(Blob params, Blob gradient);

    public void setLr(float lr) {
        this.lr = lr;
    }

    public float getLr() {
        return this.lr;
    }
}
