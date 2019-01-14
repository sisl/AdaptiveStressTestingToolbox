from mylab.algos.psmcts import PSMCTS
from mylab.optimizers.direction_constraint_optimizer import DirectionConstraintOptimizer
from garage.misc.overrides import overrides
from garage.misc import ext
from garage.tf.misc import tensor_utils
import tensorflow as tf
import numpy as np

class PSMCTSTR(PSMCTS):
	"""
	Policy Space MCTS with Trust Region Mutation
	"""
	def __init__(
			self,
			optimizer=None,
			**kwargs):
		if optimizer == None:
			self.optimizer = DirectionConstraintOptimizer()
		else:
			self.optimizer = optimizer
		super(PSMCTSTR, self).__init__(**kwargs)

	@overrides
	def init_opt(self):
		is_recurrent = int(self.policy.recurrent)
		obs_var = self.env.observation_space.new_tensor_variable(
			'obs',
			extra_dims=1 + is_recurrent,
		)
		action_var = self.env.action_space.new_tensor_variable(
			'action',
			extra_dims=1 + is_recurrent,
		)
		advantage_var = tensor_utils.new_tensor(
			'advantage',
			ndim=1 + is_recurrent,
			dtype=tf.float32,
		)

		state_info_vars = {
			k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name=k)
			for k, shape in self.policy.state_info_specs
			}
		state_info_vars_list = [state_info_vars[k] for k in self.policy.state_info_keys]

		if is_recurrent:
			valid_var = tf.placeholder(tf.float32, shape=[None, None], name="valid")

		actions = self.policy.get_action_sym(obs_var)
		if is_recurrent:
			divergence = tf.reduce_sum(tf.reduce_sum(tf.square(actions -  action_var),-1)*valid_var)/tf.reduce_sum(valid_var)
		else:
			divergence = tf.reduce_mean(tf.reduce_sum(tf.square(actions -  action_var),-1))

		input_list = [
						 obs_var,
						 action_var,
						 advantage_var,
					 ] + state_info_vars_list

		if is_recurrent:
			input_list.append(valid_var)

		self.f_divergence = tensor_utils.compile_function(
				inputs=input_list,
				outputs=divergence,
				log_name="f_divergence",
			)

		self.optimizer.update_opt(
			target=self.policy,
			# leq_constraint=(mean_kl, self.step_size), 
			leq_constraint = divergence, #input max contraint at run time with annealing
			inputs=input_list,
			constraint_name="divergence"
		)
		return dict()

	def data2inputs(self, samples_data):
		all_input_values = tuple(ext.extract(
			samples_data,
			"observations", "actions", "advantages"
		))
		agent_infos = samples_data["agent_infos"]
		state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
		all_input_values += tuple(state_info_list)
		if self.policy.recurrent:
			all_input_values += (samples_data["valids"],)
		return all_input_values

	@overrides
	def getNextAction(self,s):
		seed = np.random.randint(low= 0, high = int(2**16))
		if s.parent is None: #first generation
			# magnitude = self.initial_mag
			magnitude = self.step_size
		else:
			self.set_params(s)
			param_values = self.policy.get_param_values(trainable=True)

			# np.random.seed(seed)
			# direction = np.random.normal(size=param_values.shape)
			self.np_random.seed(seed)
			direction = self.np_random.normal(size=param_values.shape)
			
			paths = self.obtain_samples(0)
			samples_data = self.process_samples(0, paths)
			all_input_values = self.data2inputs(samples_data)
			magnitude, constraint_val = \
				self.optimizer.get_magnitude(direction=direction,inputs=all_input_values,max_constraint_val=self.step_size)
			# print("1: ",constraint_val)
			# sp,r = self.getNextState(s,(seed,magnitude))
			# self.set_params(sp)
			# divergence = self.f_divergence(*all_input_values)
			# print("2: ",divergence)
		return (seed,magnitude)
