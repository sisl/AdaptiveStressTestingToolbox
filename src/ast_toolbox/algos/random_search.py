import collections

import numpy as np
from dowel import logger
from dowel import tabular
from garage.misc import tensor_utils


# pylint: disable=too-few-public-methods
class RandomSearch:
    """Naive random search (i.e. direct Monte Carlo).

    """
    def __init__(self,
                 name="RandomSearch",
                 top_paths=None,
                 policy=None,
                 max_path_length=4,
                 discount=1,
                 batch_size=1,
                 n_itr=100,
                 **kwargs):
        self.name = name
        self.top_paths = top_paths
        self.policy = policy
        self.max_path_length = max_path_length
        self.discount = discount
        self.batch_size = batch_size
        self.n_itr = n_itr

        self._episode_reward_mean = collections.deque(maxlen=100)
        self._best_return = -np.inf

    def train(self, runner):
        """Obtain samplers and start actual training for each epoch.

        Args:
            runner (LocalRunner): LocalRunner.

        """
        for itr in runner.step_epochs():
            with logger.prefix('itr #%d | ' % itr):
                logger.log("Obtaining samples...")
                paths = runner.obtain_samples(itr)
                logger.log("Processing samples...")
                # samples_data = self.process_samples(itr, paths)

                if not (self.top_paths is None):
                    undiscounted_returns = [sum(path["rewards"]) for path in paths]
                    action_seqs = [path["actions"] for path in paths]
                    [self.top_paths.enqueue(action_seq, R, make_copy=True) for (action_seq, R) in zip(action_seqs, undiscounted_returns)]

                tabular.record('Itr', itr)
                tabular.record('StepNum', int((itr + 1) * self.batch_size))
                if self.top_paths is not None:
                    for (topi, path) in enumerate(self.top_paths):
                        tabular.record('reward ' + str(topi), path[0])

                logger.log(tabular)

    def process_samples(self, itr, paths):
        """Process sample data based on the collected paths.

        Args:
            itr (int): Iteration number.
            paths (list[dict]): A list of collected paths

        Returns:
            dict: Processed sample data, with key
                * average_return: (float)

        """
        for path in paths:
            path['returns'] = tensor_utils.discount_cumsum(
                path['rewards'], self.discount)
        average_discounted_return = (np.mean(
            [path['returns'][0] for path in paths]))
        undiscounted_returns = [sum(path['rewards']) for path in paths]
        average_return = np.mean(undiscounted_returns)

        self._episode_reward_mean.extend(undiscounted_returns)

        tabular.record('Iteration', itr)
        tabular.record('AverageDiscountedReturn', average_discounted_return)
        tabular.record('AverageReturn', average_return)
        tabular.record('Extras/EpisodeRewardMean',
                       np.mean(self._episode_reward_mean))
        tabular.record('NumTrajs', len(paths))
        tabular.record('StdReturn', np.std(undiscounted_returns))
        max_return = np.max(undiscounted_returns)
        tabular.record('MaxReturn', max_return)
        tabular.record('MinReturn', np.min(undiscounted_returns))
        if max_return > self._best_return:
            self._best_return = max_return
        tabular.record('BestReturn', self._best_return)

        return dict(average_return=average_return)
