import joblib
import tensorflow as tf

with tf.Session() as sess:
	data = joblib.load('./Data/Pong-v1/PSMCTSTRCK0.5A0.5Ec1.414Step1.0FmeanQmax/0/params.pkl')
	env = data['env']
	policy = data['policy']

	o = env.reset()
	d = False
	cr = 0.0
	while not d:
		a,_ = policy.get_action(o)
		o,r,d,_ = env.step(a)
		cr += r
	print(cr)