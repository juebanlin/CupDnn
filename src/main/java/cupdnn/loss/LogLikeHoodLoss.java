package cupdnn.loss;

import cupdnn.data.Blob;

/**
 * 对数似然损失函数
 */
public class LogLikeHoodLoss implements LossFunc {

    @Override
    public float loss(Blob label, Blob output) {
        float[] labelData = label.getData();
        float[] outputData = output.getData();
        float loss = 0.0f;
        int width = label.getWidth();
        int height = label.getHeight();
        for (int n = 0; n < label.getHeight(); n++) {
            for (int i = 0; i < label.getWidth(); i++) {
                loss -= labelData[n * width + i] * Math.log(outputData[n * width + i]);
            }
        }
        loss = loss / height;
        return loss;
    }

    @Override
    public void diff(Blob label, Blob output, Blob diff) {
        float[] labelData = label.getData();
        float[] outputData = output.getData();
        float[] diffData = diff.getData();
        int width = label.getWidth();
        int height = label.getHeight();
        diff.fillValue(0.0f);
        assert diffData.length == outputData.length : "LogLikeHoodLoss diff --- diffData.length == outputData.length error";
        assert labelData.length == outputData.length : "LogLikeHoodLoss diff --- labelData.length == outputData.length error";
        for (int n = 0; n < height; n++) {
            for (int os = 0; os < width; os++) {
                diffData[n * width + os] -= labelData[n * width + os] / outputData[n * width + os] / (float) height;
            }
        }
    }

}
