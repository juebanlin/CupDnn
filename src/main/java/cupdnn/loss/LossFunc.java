package cupdnn.loss;

import cupdnn.data.Blob;

/**
 * 损失函数
 */
public interface LossFunc {
    float loss(Blob label, Blob output);

    void diff(Blob label, Blob output, Blob diff);
}
