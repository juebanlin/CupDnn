package cupdnn.loss;

import cupdnn.data.Blob;

/**
 * 均方损失函数
 */
public class MSELoss implements LossFunc {

    @Override
    public float loss(Blob label, Blob output) {
        float[] labelData = label.getData();
        float[] outputData = output.getData();
        float loss = 0.0f;
        for (int i = 0; i < label.getSize(); ++i) {
            loss += (labelData[i] - outputData[i]) * (labelData[i] - outputData[i]);
        }
        return loss / label.getHeight();
    }

    @Override
    public void diff(Blob label, Blob output, Blob diff) {
        float[] labelData = label.getData();
        float[] outputData = output.getData();
        float[] diffData = diff.getData();
        int width = label.getWidth();
        int height = label.getHeight();
        float factor = 2;
        diff.fillValue(0.0f);
        for (int n = 0; n < height; n++) {
            for (int os = 0; os < width; os++) {
                diffData[n * width + os] += factor * (outputData[n * width + os] - labelData[n * width + os]);
            }
        }
    }
}
