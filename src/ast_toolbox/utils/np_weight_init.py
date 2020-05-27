import numpy as np
import tensorflow as tf
from garage.misc.tensor_utils import flatten_tensors
from garage.misc.tensor_utils import unflatten_tensors


# initializ parameter with numpy (easier for seeding)
# bias initialzied with0, weight initialzied with XavierUniformInitializer
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
        param_value = (np_random.rand(*shape) * 2 - 1) * init_range
    elif param.name[-3] == "b":
        param_value = np.zeros_like(param.eval())
    feed_dict = dict()
    feed_dict[policy._cached_assign_placeholders[param]] = param_value
    tf.get_default_session().run(policy._cached_assign_ops[param], feed_dict=feed_dict)


def init_policy_np(policy, np_random=np.random):
    params = policy.get_params(trainable=True)
    shapes = policy.get_param_shapes(trainable=True)
    param_values = policy.get_param_values(trainable=True)

    flattened_params = np_random.rand(*param_values.shape)
    param_values = unflatten_tensors(flattened_params, shapes)

    for i, param in enumerate(params):
        # assert param.name[-3] == "W" or param.name[-3] == "b"
        if param.name[-3] == "W":
            shape = shapes[i]
            if len(shape) == 2:
                n_inputs, n_outputs = shape
            else:
                receptive_field_size = np.prod(shape[:2])
                n_inputs = shape[-2] * receptive_field_size
                n_outputs = shape[-1] * receptive_field_size
            init_range = np.sqrt(6.0 / (n_inputs + n_outputs))
            param_values[i] = (param_values[i] * 2 - 1) * init_range
        elif param.name[-3] == "b":
            param_values[i] = np.zeros_like(param_values[i])

    param_values = flatten_tensors(param_values)
    return param_values


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
