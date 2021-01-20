package cupdnn.active;

public class ReluActivationFunc implements ActivationFunc {
	public static final String TYPE = "ReluActivationFunc";
	
	@Override
	public float active(float in) {
		return Math.max(0, in);
	}

	@Override
	public float diffActive(float in) {
		float result = in<=0 ? 0.0f:1.0f;
		return result;
	}
	
	@Override
	public String getType(){
		return TYPE;
	}

}
