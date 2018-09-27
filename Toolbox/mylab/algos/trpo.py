from sandbox.rocky.tf.algos.npo import NPO
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
import time
import rllab.misc.logger as logger
import tensorflow as tf
from rllab.sampler.utils import rollout
from rllab.misc.overrides import overrides

class TRPO(NPO):
    """
    Trust Region Policy Optimization
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            top_paths = None,
            **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = ConjugateGradientOptimizer(**optimizer_args)
        self.top_paths = top_paths
        super(TRPO, self).__init__(optimizer=optimizer, **kwargs)

    @overrides
    def train(self, sess=None, init_var=True):
        created_session = True if (sess is None) else False
        if sess is None:
            sess = tf.Session()
            sess.__enter__()
        if init_var:
            sess.run(tf.global_variables_initializer())
        self.start_worker()
        start_time = time.time()
        for itr in range(self.start_itr, self.n_itr):
            itr_start_time = time.time()
            with logger.prefix('itr #%d | ' % itr):
                logger.log("Obtaining samples...")
                paths = self.obtain_samples(itr)
                logger.log("Processing samples...")
                samples_data = self.process_samples(itr, paths)

                if not (self.top_paths is None):
                    undiscounted_returns = [sum(path["rewards"]) for path in paths]
                    action_seqs = [path["actions"] for path in paths]
                    [self.top_paths.enqueue(action_seq,R,make_copy=True) for (action_seq,R) in zip(action_seqs,undiscounted_returns)]

                logger.log("Logging diagnostics...")
                self.log_diagnostics(paths)
                logger.log("Optimizing policy...")
                self.optimize_policy(itr, samples_data)
                logger.log("Saving snapshot...")
                params = self.get_itr_snapshot(itr, samples_data)  # , **kwargs)
                if self.store_paths:
                    params["paths"] = samples_data["paths"]
                logger.save_itr_params(itr, params)
                logger.log("Saved")
                logger.record_tabular('Time', time.time() - start_time)
                logger.record_tabular('ItrTime', time.time() - itr_start_time)
                logger.dump_tabular(with_prefix=False)
                if self.plot:
                    rollout(self.env, self.policy, animated=True, max_path_length=self.max_path_length)
                    if self.pause_for_plot:
                        input("Plotting evaluation run: Press Enter to "
                              "continue...")
        self.shutdown_worker()
        if created_session:
            sess.close()