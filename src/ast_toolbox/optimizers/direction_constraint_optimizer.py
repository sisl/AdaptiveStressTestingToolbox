import numpy as np
from dowel import logger
from garage.tf.misc import tensor_utils
from garage.tf.optimizers.conjugate_gradient_optimizer import PearlmutterHvp
from garage.tf.optimizers.utils import LazyDict
from garage.tf.optimizers.utils import sliced_fun


class DirectionConstraintOptimizer:
    """Performs constrained optimization via line search on the given gradient direction.

    Parameters
    ----------
    cg_iters : int, optional
        The number of CG iterations used to calculate A^-1 g.
    reg_coeff : float, optional
        A small value so that A -> A + reg*I.
    subsample_factor : int, optional
        Subsampling factor to reduce samples when using "conjugate gradient. Since the
        computation time for the descent direction dominates, this can greatly reduce the overall computation time.
    debug_nan : bool, optional
        If set to True, NanGuard will be added to the compilation, and ipdb will be invoked when
        nan is detected.
    accept_violation : bool, optional
        Whether to accept the descent step if it violates the line search condition after
        exhausting all backtracking budgets.
    """

    def __init__(
            self,
            cg_iters=10,
            reg_coeff=1e-5,
            subsample_factor=1.,
            backtrack_ratio=0.8,
            max_backtracks=15,
            debug_nan=False,
            accept_violation=False,
            hvp_approach=None,
            num_slices=1):
        self._cg_iters = cg_iters
        self._reg_coeff = reg_coeff
        self._subsample_factor = subsample_factor
        self._backtrack_ratio = backtrack_ratio
        self._max_backtracks = max_backtracks
        self._num_slices = num_slices

        self._opt_fun = None
        self._target = None
        self._max_constraint_val = None
        self._constraint_name = None
        self._debug_nan = debug_nan
        self._accept_violation = accept_violation
        if hvp_approach is None:
            hvp_approach = PearlmutterHvp(num_slices)
        self._hvp_approach = hvp_approach

    def update_opt(self, target, leq_constraint, inputs, extra_inputs=None, constraint_name="constraint", *args,
                   **kwargs):
        """Update the internal tensowflow operations.

        Parameters
        ----------
        target :
            A parameterized object to optimize over. It should implement methods of the
            :py:class:`garage.core.paramerized.Parameterized` class.
        leq_constraint : :py:class:'tensorflow.Tensor'
            The variable to be constrained.
        inputs :
            A list of symbolic variables as inputs, which could be subsampled if needed. It is assumed
            that the first dimension of these inputs should correspond to the number of data points.
        extra_inputs :
            A list of symbolic variables as extra inputs which should not be subsampled.
        """

        inputs = tuple(inputs)
        if extra_inputs is None:
            extra_inputs = tuple()
        else:
            extra_inputs = tuple(extra_inputs)

        # constraint_term, constraint_value = leq_constraint
        constraint_term = leq_constraint

        # params = target.get_params(trainable=True)

        self._hvp_approach.update_hvp(f=constraint_term, target=target, inputs=inputs + extra_inputs,
                                      reg_coeff=self._reg_coeff)

        self._target = target
        # self._max_constraint_val = constraint_value
        self._max_constraint_val = np.inf
        self._constraint_name = constraint_name

        self._opt_fun = LazyDict(
            f_constraint=lambda: tensor_utils.compile_function(
                inputs=inputs + extra_inputs,
                outputs=constraint_term,
                log_name="constraint",
            ),
        )

    def constraint_val(self, inputs, extra_inputs=None):
        """Calculate the constraint value.

        Parameters
        ----------
        inputs :
            A list of symbolic variables as inputs, which could be subsampled if needed. It is assumed
            that the first dimension of these inputs should correspond to the number of data points.
        extra_inputs : optional
            A list of symbolic variables as extra inputs which should not be subsampled.

        Returns
        -------
        constraint_value : float
            The value of the constrained variable.
        """
        inputs = tuple(inputs)
        if extra_inputs is None:
            extra_inputs = tuple()
        return sliced_fun(self._opt_fun["f_constraint"], self._num_slices)(inputs, extra_inputs)

    def get_magnitude(self, direction, inputs, max_constraint_val=None, extra_inputs=None, subsample_grouped_inputs=None):
        """Calculate the update magnitude.

        Parameters
        ----------
        direction: :py:class:'tensorflow.Tensor'
            The gradient direction.
        inputs :
            A list of symbolic variables as inputs, which could be subsampled if needed. It is assumed
            that the first dimension of these inputs should correspond to the number of data points.
        max_constraint_val : float, optional
            The maximum value for the constrained variale.
        extra_inputs : optional
            A list of symbolic variables as extra inputs which should not be subsampled.
        subsample_grouped_inputs : optional
            The list of inputs that are needed to be subsampled.

        Returns
        -------
        magnitude : float
            The update magnitude.
        """
        if max_constraint_val is not None:
            self._max_constraint_val = max_constraint_val
        prev_param = np.copy(self._target.get_param_values(trainable=True))
        inputs = tuple(inputs)
        if extra_inputs is None:
            extra_inputs = tuple()

        if self._subsample_factor < 1:
            if subsample_grouped_inputs is None:
                subsample_grouped_inputs = [inputs]
            subsample_inputs = tuple()
            for inputs_grouped in subsample_grouped_inputs:
                n_samples = len(inputs_grouped[0])
                inds = np.random.choice(
                    n_samples, int(n_samples * self._subsample_factor), replace=False)
                subsample_inputs += tuple([x[inds] for x in inputs_grouped])
        else:
            subsample_inputs = inputs

        Hx = self._hvp_approach.build_eval(subsample_inputs + extra_inputs)

        descent_direction = direction

        initial_step_size = np.sqrt(
            2.0 * self._max_constraint_val * (1. / (descent_direction.dot(Hx(descent_direction)) + 1e-8))
        )
        if np.isnan(initial_step_size):
            initial_step_size = 1.
        flat_descent_step = initial_step_size * descent_direction

        n_iter = 0
        for n_iter, ratio in enumerate(self._backtrack_ratio ** np.arange(self._max_backtracks)):
            cur_step = ratio * flat_descent_step
            cur_param = prev_param - cur_step
            self._target.set_param_values(cur_param, trainable=True)
            constraint_val = sliced_fun(self._opt_fun["f_constraint"], self._num_slices)(inputs, extra_inputs)
            if self._debug_nan and np.isnan(constraint_val):
                import ipdb
                ipdb.set_trace()
            if constraint_val <= self._max_constraint_val:
                break
        if (np.isnan(constraint_val) or constraint_val >= self._max_constraint_val) and not self._accept_violation:
            logger.log("Line search condition violated. Rejecting the step!")
            if np.isnan(constraint_val):
                logger.log("Violated because constraint %s is NaN" % self._constraint_name)
            if constraint_val >= self._max_constraint_val:
                logger.log("Violated because constraint %s is violated" % self._constraint_name)
            self._target.set_param_values(prev_param, trainable=True)
        # logger.log("backtrack iters: %d" % n_iter)
        # logger.log("final magnitude: " + str(-ratio*initial_step_size))
        logger.log("final kl: " + str(constraint_val))
        # logger.log("optimization finished")
        return -ratio * initial_step_size, constraint_val

    def __getstate__(self):
        """Get the internal state.

        Returns
        -------
        data : dict
            The intertal state dict.
        """
        new_dict = self.__dict__.copy()
        del new_dict['_opt_fun']
        return new_dict
