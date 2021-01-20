package cupdnn.active;

/**
 * 激活函数
 */
public  interface ActivationFunc {
	float active(float in);
	float diffActive(float in);
	String getType();
}
