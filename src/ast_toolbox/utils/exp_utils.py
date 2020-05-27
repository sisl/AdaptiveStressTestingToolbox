import numpy as np


def log_mean_exp(x, dim):
    """
    Compute the log(mean(exp(x), dim)) in a numerically stable manner
    """
    return log_sum_exp(x, dim) - np.log(x.shape[dim])


def log_sum_exp(x, dim):
    """
    Compute the log(sum(exp(x), dim)) in a numerically stable manner
    """
    max_x = np.max(x, dim)
    new_x = x - np.repeat(np.expand_dims(max_x, dim), x.shape[dim], dim)
    return max_x + np.log(np.sum(np.exp(new_x), dim))


def softmax(x, dim):
    """Compute softmax values for each sets of scores in x along dim"""
    e_x = np.exp(x - np.max(x, dim))
    return e_x / np.sum(e_x, dim)
