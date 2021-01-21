package cupdnn.util;

import java.util.Collection;
import java.util.concurrent.*;

public class ThreadPoolManager {
    private static ThreadPoolManager instance;

    public static ThreadPoolManager getInstance(int threadNum) {
        synchronized (ThreadPoolManager.class) {
            if (instance == null) {
                instance = new ThreadPoolManager(threadNum);
            }
        }
        return instance;
    }

    //1、配置线程池
    private ThreadPoolExecutor threadPool;

    private ThreadPoolManager(int threadNum) {
        if (threadNum < 1) {
            threadNum = 4;
        }
        threadPool = new ThreadPoolExecutor(threadNum, threadNum, 1000, TimeUnit.SECONDS, new LinkedBlockingQueue<>());
    }

    /**
     * 执行所有任务并等待完成
     * @param tasks
     */
    public void runTasks(Collection<Runnable> tasks){
        CountDownLatch countDownLatch=new CountDownLatch(tasks.size());
        for (Runnable task : tasks) {
            threadPool.execute(()->{
                try {
                    task.run();
                }catch (Throwable e){

                }
                finally {
                    countDownLatch.countDown();
                }
            });
        }
        try {
            countDownLatch.await();
        }catch (Exception e){
            e.printStackTrace();
        }
    }
}
