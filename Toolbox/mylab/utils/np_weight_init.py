import numpy as np
import tensorflow as tf

#initializ parameter with numpy (easier for seeding)
#bias initialzied with0, weight initialzied with XavierUniformInitializer
def init_param_np(param, policy, np_random=np.random):
    assert param.name[-3] == "W" or param.name[-3] == "b"
    if param.name[-3] == "W":
        shape = tf.shape(param).eval()
        if len(shape) == 2:
            n_inputs, n_outputs = shape
        else:
            receptive_field_size = np.prod(shape[:2])
            n_inputs = shape[-2] * receptive_field_size
            n_outputs = shape[-1] * receptive_field_size
        init_range = np.sqrt(6.0 / (n_inputs + n_outputs))
        # return tf.random_uniform_initializer(-init_range, init_range, dtype=dtype)(shape)
        param_value = (np_random.rand(*shape)*2-1)*init_range
    elif param.name[-3] == "b":
        param_value = np.zeros_like(param.eval())
    feed_dict = dict()
    feed_dict[policy._cached_assign_placeholders[param]] = param_value
    tf.get_default_session().run(policy._cached_assign_ops[param], feed_dict=feed_dict)

# class XavierUniformInitializer(object):
#     def __call__(self, shape, dtype=tf.float32, *args, **kwargs):
#         if len(shape) == 2:
#             n_inputs, n_outputs = shape
#         else:
#             receptive_field_size = np.prod(shape[:2])
#             n_inputs = shape[-2] * receptive_field_size
#             n_outputs = shape[-1] * receptive_field_size
#         init_range = math.sqrt(6.0 / (n_inputs + n_outputs))
#         return tf.random_uniform_initializer(-init_range, init_range, dtype=dtype)(shape)


# class HeUniformInitializer(object):
#     def __call__(self, shape, dtype=tf.float32, *args, **kwargs):
#         if len(shape) == 2:
#             n_inputs, _ = shape
#         else:
#             receptive_field_size = np.prod(shape[:2])
#             n_inputs = shape[-2] * receptive_field_size
#         init_range = math.sqrt(1.0 / n_inputs)
#         return tf.random_uniform_initializer(-init_range, init_range, dtype=dtype)(shape)