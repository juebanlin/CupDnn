package cupdnn.active;

public class TanhActivationFunc implements ActivationFunc{
	public static final String TYPE = "TanhActivationFunc";

	private float tanh(float in){
		float ef = (float) Math.exp(in);
		float efx = (float) Math.exp(-in);
		return (ef-efx)/(ef+efx);
	}
	@Override
	public float active(float in) {
		return tanh(in);
	}

	@Override
	public float diffActive(float in) {
		return (1-tanh(in)*tanh(in));
	}
	
	@Override
	public String getType(){
		return TYPE;
	}

}
