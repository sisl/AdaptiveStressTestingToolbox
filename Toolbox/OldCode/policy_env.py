from rllab.envs.base import Env
from rllab.envs.base import Step
import numpy as np
import pdb
import pickle as pickle
from sandbox.rocky.tf.misc import tensor_utils

class PolicyEnv(Env):
    def __init__(self,
                 interactive=True,
                 sample_init_state=False,
                 s_0=None,
                 simulator=None,
                 reward_function=None,
                 spaces=None,
                 vectorized=False):
        # Constant hyper-params -- set by user
        self.interactive = interactive
        self.spaces = spaces
        # These are set by reset, not the user
        self._done = False
        self._reward = 0.0
        self._info = []
        self._step = 0
        self._action = None
        self._actions = []
        self._first_step = True

        if s_0 is None:
            self._init_state = self.observation_space.sample()
        else:
            self._init_state = s_0
        self._sample_init_state = sample_init_state
        self.simulator = simulator
        if self.simulator is None:
            self.simulator = ExampleAVSimulator()
        self.reward_function = reward_function
        self.spaces = spaces

        self.vectorized = vectorized
        self.step_count = 0 # number of setp call in training, not useful if vecterized=True

        super().__init__()

    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of episode
        is reached, reset() should be called to reset the environment's internal state.
        Input
        -----
        action : an action provided by the environment
        Outputs
        -------
        (observation, reward, done, info)
        observation : agent's observation of the current environment
        reward [Float] : amount of reward due to the previous action
        done : a boolean, indicating whether the episode has ended
        info : a dictionary containing other diagnostic information from the previous action
        """
        self._action = action
        self._actions.append(action)
        # Update simulation step
        obs = self.simulator.step(self._action)
        if not self.interactive:
            obs = self._init_state
        # if self.simulator.is_goal():
        if self.simulator.isterminal():
            self._done = True
        # Calculate the reward for this step
        self._reward = self.reward_function.give_reward(
            action=self._action,
            info=self.simulator.get_reward_info())
        # Update instance attributes
        # self.log()
        # if self._step == self.c_max_path_length - 1:
        #     # pdb.set_trace()
        #     self.simulator.simulate(self._actions)
        self._step = self._step + 1

        self.step_count += 1

        return Step(observation=obs,
                    reward=self._reward,
                    done=self._done,
                    info={'cache': self._info})

    def simulate(self, actions):
        if self._sample_init_state:
            self._init_state = self.observation_space.sample()
        self.simulator.simulate(actions, self._init_state)

    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        self._actions = []
        if self._sample_init_state:
            self._init_state = self.observation_space.sample()
        self._done = False
        self._reward = 0.0
        self._info = []
        self._step = 0
        self._action = None
        self._actions = []
        self._first_step = True

        return self.simulator.reset(self._init_state)
    def seed(self,seed):
        return self.simulator.seed(seed)

    @property
    def action_space(self):
        """
        Returns a Space object
        """
        return self.spaces.action_space

    @property
    def observation_space(self):
        """
        Returns a Space object
        """

        return self.spaces.observation_space

    def get_cache_list(self):
        return self._info

    def log(self):
        self.simulator.log()

    def render(self):
        if hasattr(self.simulator, "render") and callable(getattr(self.simulator, "render")):
            return self.simulator.render()
        else:
            return None

    def close(self):
        if hasattr(self.simulator, "close") and callable(getattr(self.simulator, "close")):
            self.simulator.close()
        else:
            return None

    def vec_env_executor(self, n_envs, max_path_length):
        envs = [pickle.loads(pickle.dumps(self.simulator.env)) for _ in range(n_envs)]
        return InnerVecEnvExecutor(envs, self.simulator.policy, self.reward_function,
                    self._sample_init_state, self._init_state,
                    max_path_length, self.interactive)

class InnerVecEnvExecutor(object):
    def __init__(self, envs, policy, reward_function, sample_init_state, init_state, max_path_length, interactive):
        self.envs = envs
        self._action_space = envs[0].ast_action_space
        self._observation_space = envs[0].ast_observation_space
        self.ts = np.zeros(len(self.envs), dtype='int')
        self.max_path_length = max_path_length
        self.policy = policy
        self.reward_function = reward_function
        self._sample_init_state = sample_init_state
        self._init_state = init_state
        self.interactive = interactive

    def step(self, action_n):

        self.ts += 1

        ast_action_n = action_n
        os = [np.reshape(env.get_observation(),env.ast_observation_space.shape) for env in self.envs]
        action_n, action_info_n = self.policy.get_actions(os)
        if "mean" in action_info_n:
            action_n = action_info_n["mean"]
        elif "prob" in action_info_n:
            action_n = np.argmax(action_info_n["prob"],axis=1)
        # action = self.env.action_space.sample()
        # results = [np.reshape(env.ast_step(action, ast_action),env.ast_observation_space.shape) for (action,ast_action,env) in zip(action_n, ast_action_n, self.envs)]
        results = [env.ast_step(action, ast_action) for (action,ast_action,env) in zip(action_n, ast_action_n, self.envs)]
        if self.interactive:
            obs = [np.reshape(ob,env.ast_observation_space.shape) for (ob,env) in zip(list(zip(*results))[0],self.envs)]
        else:
            obs = [self._init_state for env in self.envs]
        obs = np.asarray(obs)
        dones = list(zip(*results))[2]
        dones = np.asarray(dones)
        if self.max_path_length is not None:
            dones[self.ts >= self.max_path_length] = True

        infos = [env.ast_get_reward_info() for env in self.envs]
        for (i,info) in enumerate(infos):
            info['is_terminal'] = dones[i]
        rewards = [self.reward_function.give_reward(action=action,info=info)\
                    for (action,info) in zip(ast_action_n, infos)]
        env_infos = infos

        rewards = np.asarray(rewards)
        
        for (i, done) in enumerate(dones):
            if done:
                if self._sample_init_state:
                    obs[i] = self.envs[i].ast_reset(self.observation_space.sample())[0]
                else:
                    obs[i] = self.envs[i].ast_reset(self._init_state)[0]
                self.ts[i] = 0
        return obs, rewards, dones, tensor_utils.stack_tensor_dict_list(env_infos)

    def reset(self):
        if self._sample_init_state:
            results = [env.ast_reset(self.observation_space.sample())[0] for env in self.envs]
        else:
            results = [env.ast_reset(self._init_state)[0] for env in self.envs]
        self.ts[:] = 0
        return results

    @property
    def num_envs(self):
        return len(self.envs)

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def terminate(self):
        pass

