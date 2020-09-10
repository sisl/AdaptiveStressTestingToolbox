"""Class to define the action and observation spaces of an AST problem."""


class ASTSpaces(object):
    """Class to define the action and observation spaces of an AST problem.

    Both the `action_space` and the `observation_space` should be a
    `gym.spaces.Space <https://gym.openai.com/docs/#spaces>`_ type.

    The `action_space` is only used to clip actions if `ASTEnv` is wrapped by the normalize env.

    If using `ASTEnv` with `blackbox_sim_state == True`, `observation_space` should define the space for each
    simulation state variable. Otherwise, it should define the space of initial condition variables.

    If using `ASTEnv` with `fixed_init_state == False`, the initial conditions of each rollout will be randomly
    sampled at uniform from the observation_space.

    """

    def __init__(self):
        pass

    @property
    def action_space(self):
        """Returns a definition of the action space of the reinforcement learning problem.

        Returns
        -------
        : `gym.spaces.Space <https://gym.openai.com/docs/#spaces>`_
            The action space of the reinforcement learning problem.
        """
        raise NotImplementedError

    @property
    def observation_space(self):
        """Returns a definition of the observation space of the reinforcement learning problem.

        Returns
        -------
        : `gym.spaces.Space <https://gym.openai.com/docs/#spaces>`_
            The observation space of the reinforcement learning problem.
        """
        raise NotImplementedError
