package test.mnist;

import cupdnn.Network;
import cupdnn.active.ReluActivationFunc;
import cupdnn.layer.*;
import cupdnn.loss.MSELoss;
import cupdnn.optimizer.SGDOptimizer;
import cupdnn.util.DigitImage;

import java.util.List;

public class MnistNetwork {
	Network network;
	/**
	 * 构建全连接网络
	 */
	private void buildFullConnNetwork(){
		//给network添加网络层
		InputLayer layer1 = new InputLayer(network,28,28,1);
		network.addLayer(layer1);
		FullConnectionLayer layer2 = new FullConnectionLayer(network,28*28,512);
		layer2.setActivationFunc(new ReluActivationFunc());
		network.addLayer(layer2);
		FullConnectionLayer layer3 = new FullConnectionLayer(network,512,64);
		layer3.setActivationFunc(new ReluActivationFunc());
		network.addLayer(layer3);
		FullConnectionLayer layer4 = new FullConnectionLayer(network,64,10);
		layer4.setActivationFunc(new ReluActivationFunc());
		network.addLayer(layer4);
		SoftMaxLayer sflayer = new SoftMaxLayer(network,10);
		network.addLayer(sflayer);
	}

	/**
	 * 构建卷积网络
	 */
	private void buildConvNetwork(){
		InputLayer layer1 =  new InputLayer(network,28,28,1);
		network.addLayer(layer1);
		//输入1通道,输出6通道(6个卷积核)
		//卷积核通道数等于数据图像通道数,卷积核输出层通道数等于卷积核个数
		Conv2dLayer conv1 = new Conv2dLayer(network,28,28,1,6,3,1);
		conv1.setActivationFunc(new ReluActivationFunc());
		network.addLayer(conv1);
		//(inputSize-kernelSize)/stride+1=outSize ==>  (28-2)/2+1=14
		PoolMaxLayer pool1 = new PoolMaxLayer(network,14,14,6,2,2);
		network.addLayer(pool1);
		
		Conv2dLayer conv2 = new Conv2dLayer(network,14,14,6,6,3,1);
		conv2.setActivationFunc(new ReluActivationFunc());
		network.addLayer(conv2);
		//(inputSize-kernelSize)/stride+1=outSize ==> (14-2)/2+1=7
		PoolMeanLayer pool2 = new PoolMeanLayer(network,7,7,6,2,2);
		network.addLayer(pool2);
	
		FullConnectionLayer fc1 = new FullConnectionLayer(network,7*7*6,256);
		fc1.setActivationFunc(new ReluActivationFunc());
		network.addLayer(fc1);
		
		FullConnectionLayer fc2 = new FullConnectionLayer(network,256,10);
		fc2.setActivationFunc(new ReluActivationFunc());
		network.addLayer(fc2);
		
		SoftMaxLayer sflayer = new SoftMaxLayer(network,10);
		network.addLayer(sflayer);
	}

	public void buildNetwork(){
		//首先构建神经网络对象，并设置参数
		network = new Network();
		network.setThreadNum(16);
		network.setBatch(20);
		network.setLrDecay(0.9f);
		//network.setLoss(new LogLikeHoodLoss());
		//network.setLoss(new CrossEntropyLoss());
		network.setLoss(new MSELoss());
		network.setOptimizer(new SGDOptimizer(0.1f));
		//buildFullConnNetwork();
		buildConvNetwork();
		network.prepare();
	}
	
	public void train(List<DigitImage> trainLists,int epoes,List<DigitImage> testLists) {
		network.train(trainLists, epoes, testLists);
	}
	
	public void test(List<DigitImage> imgList) {
		network.test(imgList);
	}

	
	public void saveModel(String path){
		network.saveModel(path);
	}
	
	public void loadModel(String path){
		network = new Network();
		network.loadModel(path);
		network.prepare();
	}
}
