package test.mnist;

import java.io.IOException;
import java.net.URL;
import java.util.List;

import cupdnn.util.DigitImage;
import test.mnist.MnistNetwork;
import test.mnist.ReadFile;


public class MnistTest {
    static List<DigitImage> trains = null;
    static List<DigitImage> tests = null;

    public static void main(String[] args) {
        //load mnist
        ClassLoader classLoader = MnistTest.class.getClassLoader();

        String path1 = classLoader.getResource("data/mnist/train-labels.idx1-ubyte").getPath();
        String path2 = classLoader.getResource("data/mnist/train-images.idx3-ubyte").getPath();
        String path3 = classLoader.getResource("data/mnist/t10k-labels.idx1-ubyte").getPath();
        String path4 = classLoader.getResource("data/mnist/t10k-images.idx3-ubyte").getPath();
        String modelPath= classLoader.getResource("model/mnist.model").getPath();

        ReadFile rf1 = new ReadFile(path1, path2);
        ReadFile rf2 = new ReadFile(path3, path4);
        try {
            tests = rf2.loadDigitImages();
            trains = rf1.loadDigitImages();
        } catch (IOException e) {
            e.printStackTrace();
        }
        MnistNetwork mn = new MnistNetwork();
        mn.buildNetwork();
        mn.train(trains, 30, tests);
        mn.saveModel(modelPath);
        mn.loadModel(modelPath);
        mn.test(tests);
    }
}
