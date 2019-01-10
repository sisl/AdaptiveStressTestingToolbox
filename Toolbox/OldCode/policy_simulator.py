from mylab.simulators.ast_simulator import ASTSimulator
import numpy as np

#Define the class
class PolicySimulator(ASTSimulator):
    """
    Class template for a non-interactive simulator.
    """
    #Accept parameters for defining the behavior of the system under test[SUT]
    def __init__(self,
    			 env,
    			 policy,
                 **kwargs):

        #initialize the base Simulator
        self.env = env
        self.policy = policy
        self.path_length = 0
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
            action, agent_info = self.policy.get_action(o)
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
        action, agent_info = self.policy.get_action(o)
        if "mean" in agent_info:
            action = agent_info["mean"]
        elif "prob" in agent_info:
            action = np.argmax(agent_info["prob"])
        # action = self.env.action_space.sample()
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
        o_ast, o = self.env.ast_reset(s_0)
        self.path_length = 0
        self._is_terminal = False
        return o_ast

    def is_goal(self):
        return self.env.ast_is_goal()


    def get_reward_info(self):
        """
        returns any info needed by the reward function to calculate the current reward
        """
        info = self.env.ast_get_reward_info()
        info["is_terminal"] = self._is_terminal
        return info

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

    def isterminal(self):
        return self._is_terminal

    def seed(self,seed):
        return self.env.seed(seed)

    @property
    def observation_space(self):
        return self.env.ast_observation_space

    @property
    def action_space(self):
        return self.env.ast_action_space