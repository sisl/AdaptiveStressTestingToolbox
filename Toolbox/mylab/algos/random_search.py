from enum import Enum
from enum import unique

import numpy as np
import tensorflow as tf
import time
from garage.misc import logger
from garage.misc import special
from garage.tf.algos import BatchPolopt
from garage.tf.misc import tensor_utils

class RandomSearch(BatchPolopt):
    def __init__(self,
                 name="RandomSearch",
                 top_paths=None,
                 policy=None,
                 **kwargs):
        self.name = name
        self.top_paths = top_paths
        super(RandomSearch, self).__init__(policy=policy, **kwargs)

    def init_opt(self):
        return dict()

    def optimize_policy(self, itr, samples_data):
        num_traj = self.batch_size // self.max_path_length
        actions = samples_data["actions"][:num_traj, ...]
        logger.record_histogram("{}/Actions".format(self.policy.name), actions)

    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            # baseline=self.baseline,
            env=self.env,
        )

    def train(self, sess=None, init_var=False):
        created_session = True if (sess is None) else False
        if sess is None:
            sess = tf.Session()
            sess.__enter__()
        if init_var:
            sess.run(tf.global_variables_initializer())
        self.start_worker(sess)
        start_time = time.time()
        for itr in range(self.start_itr, self.n_itr):
            itr_start_time = time.time()
            with logger.prefix('itr #%d | ' % itr):
                logger.log("Obtaining samples...")
                paths = self.obtain_samples(itr)
                logger.log("Processing samples...")
                samples_data = self.process_samples(itr, paths)
                # print([path["observations"] for path in paths])

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
                # logger.record_tabular('Time', time.time() - start_time)
                # logger.record_tabular('ItrTime', time.time() - itr_start_time)
                logger.record_tabular('Itr',itr)
                logger.record_tabular('StepNum',int((itr+1)*self.batch_size))
                if self.top_paths is not None:
                    for (topi, path) in enumerate(self.top_paths):
                        logger.record_tabular('reward '+str(topi), path[0])

                logger.dump_tabular(with_prefix=False)
                if self.plot:
                    self.plotter.update_plot(self.policy, self.max_path_length)
                    if self.pause_for_plot:
                        input("Plotting evaluation run: Press Enter to "
                              "continue...")
        self.shutdown_worker()
        if created_session:
            sess.close()