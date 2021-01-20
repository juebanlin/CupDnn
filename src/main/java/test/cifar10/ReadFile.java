package test.cifar10;

import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import cupdnn.util.DigitImage;


public class ReadFile {

    private String fileName;
    private final int picSize = 32 * 32 * 3;
    public List<DigitImage> images;


    public ReadFile(String fileName) {
        this.fileName = fileName;
    }

    public List<DigitImage> loadDigitImages(boolean isTrain) throws IOException {
        images = new ArrayList<>();
        if (isTrain) {
            for (int i = 1; i < 6; i++) {
                String tmpFileName = fileName.replace('%', (char) ('0' + i));
                //System.out.println(tmpFileName);
                InputStream fileInputStream = getClass().getClassLoader().getResourceAsStream(tmpFileName);
                for (int j = 0; j < 10000; j++) {
                    int label = fileInputStream.read();
                    //System.out.println("label: "+label);
                    byte[] buffer = new byte[picSize];  //32*32*3
                    fileInputStream.read(buffer);
                    DigitImage digitImage = new DigitImage(label, buffer);
                    images.add(digitImage);
                }
                fileInputStream.close();
            }
        } else {
            InputStream fileInputStream = getClass().getClassLoader().getResourceAsStream(fileName);
            for (int j = 0; j < 10000; j++) {
                int label = fileInputStream.read();
                //System.out.println("label: "+label);
                byte[] buffer = new byte[picSize];  //32*32*3
                fileInputStream.read(buffer);
                DigitImage digitImage = new DigitImage(label, buffer);
                images.add(digitImage);
            }
            fileInputStream.close();
        }
        return images;
    }
}
