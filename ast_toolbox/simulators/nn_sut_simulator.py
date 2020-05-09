from ast_toolbox.simulators.ast_simulator import ASTSimulator
import numpy as np
import pickle as pickle
from garage.tf.misc import tensor_utils

#Define the class
class NNSUTSimulator(ASTSimulator):
    """
    neural network system under test simulator
    """

    def __init__(self,
    			 env,
    			 sut, #system under test, in this case is a NN policy
                 **kwargs):

        #initialize the base Simulator
        self.env = env
        self.sut = sut
        self.path_length = 0
        
        self._is_terminal = False
        super().__init__(**kwargs)

    def simulate(self, actions, s_0):
        """
        Run/finish the simulation
        Input
        -----
        action : A sequential list of actions taken by the simulation
        Outputs
        -------
        (terminal_index)
        terminal_index : The index of the action that resulted in a state in the goal set E. If no state is found
                        terminal_index should be returned as -1.

        """
        # initialize the simulation
        path_length = 0
        done = False
        o_ast, o = self.env.ast_reset(s_0)
        self._info  = []

        # Take simulation steps unbtil horizon is reached
        while (path_length < self.c_max_path_length) and (not done):
            #get the action from the list
            ast_action = actions[path_length]
            action, agent_info = self.sut.get_action(o)
            o_ast, o, done = self.env.ast_step(action, ast_action)

            # check if a crash has occurred. If so return the timestep, otherwise continue
            if self.env.ast_is_goal():
                return path_length, np.array(self._info)
            path_length = path_length + 1
            self._is_terminal = (self.path_length >= self.c_max_path_length) or done
        # horizon reached without crash, return -1
        return -1, np.array(self._info)

    def step(self, action):
        ast_action = action
        o = self.env.get_observation()
        action, agent_info = self.sut.get_action(o)
        if "mean" in agent_info:
            action = agent_info["mean"]
        elif "prob" in agent_info:
            action = np.argmax(agent_info["prob"])
        if self.sut.recurrent:
            self.sut.prev_actions = self.sut.action_space.flatten_n([action])
        o_ast, o, done = self.env.ast_step(action, ast_action)
        self.path_length += 1
        self._is_terminal = (self.path_length >= self.c_max_path_length) or done
        return o_ast

    def reset(self, s_0):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        self.sut.reset()
        o_ast, o = self.env.ast_reset(s_0)
        self.path_length = 0
        self._is_terminal = False
        return o_ast

    def get_reward_info(self):
        """
        returns any info needed by the reward function to calculate the current reward
        """
        info = self.env.ast_get_reward_info()
        info["is_terminal"] = self._is_terminal
        return info

    def is_goal(self):
        return self.env.ast_is_goal()

    def log(self):
        if hasattr(self.env, "log") and callable(getattr(self.env, "log")):
            return self.env.log()
        else:
            return None

    def render(self):
        if hasattr(self.env, "render") and callable(getattr(self.env, "render")):
            return self.env.render()
        else:
            return None

    def close(self):
        if hasattr(self.env, "close") and callable(getattr(self.env, "close")):
            self.env.close()
        else:
            return None

    def seed(self,seed):
        return self.env.seed(seed)

    @property
    def observation_space(self):
        return self.env.ast_observation_space

    @property
    def action_space(self):
        return self.env.ast_action_space

    def vec_env_executor(self, n_envs, max_path_length, reward_function,
                            fixed_init_state, init_state, open_loop):
        envs = [pickle.loads(pickle.dumps(self.env)) for _ in range(n_envs)]
        return InnerVecEnvExecutor(envs, self.sut, reward_function,
                    fixed_init_state, init_state,
                    max_path_length, open_loop)

class InnerVecEnvExecutor(object):
    def __init__(self, envs, sut, reward_function, fixed_init_state, init_state, max_path_length, open_loop):
        self.envs = envs
        self._action_space = envs[0].ast_action_space
        self._observation_space = envs[0].ast_observation_space
        self.ts = np.zeros(len(self.envs), dtype='int')
        self.max_path_length = max_path_length
        self.sut = sut
        self.reward_function = reward_function
        self._fixed_init_state = fixed_init_state
        self._init_state = init_state
        self.open_loop = open_loop

    def step(self, action_n):
        self.ts += 1

        ast_action_n = action_n
        os = [np.reshape(env.get_observation(),env.observation_space.shape) for env in self.envs]
        action_n, action_info_n = self.sut.get_actions(os)
        if "mean" in action_info_n:
            action_n = action_info_n["mean"]
        elif "prob" in action_info_n:
            action_n = np.argmax(action_info_n["prob"],axis=1)
        if self.sut.recurrent:
            self.sut.prev_actions = self.sut.action_space.flatten_n(action_n)
        # action = self.env.action_space.sample()
        # results = [np.reshape(env.ast_step(action, ast_action),env.ast_observation_space.shape) for (action,ast_action,env) in zip(action_n, ast_action_n, self.envs)]
        results = [env.ast_step(action, ast_action) for (action,ast_action,env) in zip(action_n, ast_action_n, self.envs)]
        if self.open_loop:
            obs = [self._init_state for env in self.envs]
        else:
            obs = [np.reshape(ob,env.ast_observation_space.shape) for (ob,env) in zip(list(zip(*results))[0],self.envs)]

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
                if self._fixed_init_state:
                    obs[i] = self.envs[i].ast_reset(self._init_state)[0]
                else:
                    obs[i] = self.envs[i].ast_reset(self.observation_space.sample())[0]
                self.ts[i] = 0
        self.sut.reset(dones)
        return obs, rewards, dones, tensor_utils.stack_tensor_dict_list(env_infos)

    def reset(self):
        if self._fixed_init_state:
            results = [env.ast_reset(self._init_state)[0] for env in self.envs]
        else:
            results = [env.ast_reset(self.observation_space.sample())[0] for env in self.envs]
        self.ts[:] = 0
        dones = np.asarray([True] * len(self.envs))
        self.sut.reset(dones)
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

    def close(self):
        pass