import time

# def identity(*args):
#     if len(args) == 1:
#         return args[0]
#     return args


class TransitionModel:
    """The wrapper for the transitin model used in the tree search.

    Parameters
    ----------
    getInitialState : function
        getInitialState() returns the initial AST state.
    getNextState : function
        getNextState(s, a) returns the next state and the reward.
    isEndState : function
        isEndState(s) returns whether s is a terminal state.
    maxSteps : int
        The maximum path length.
    goToState : function
        goToState(s) sets the simulator to the target state s.
    """

    def __init__(self, getInitialState, getNextState, isEndState, maxSteps, goToState):
        self.getInitialState = getInitialState
        self.getNextState = getNextState
        self.isEndState = isEndState
        self.maxSteps = maxSteps
        self.goToState = goToState


def simulate(model, p, policy, verbose=False, sleeptime=0.0):
    """Simulate the environment model using the policy and the parameter p.

    Parameters
    ----------
    model : :py:class:`ast_toolbox.mcts.MDP.TransitionModel`
        The environment model.
    p :
        The extra paramters needed by the policy.
    policy : function
        policy(p, s) returns the next action.
    verbose : bool, optional
        Whether to logging simulating information.
    sleeptime: float, optional
        The pause time between each step.

    Returns
    ----------
    cum_reward : float
        The cumulative reward.
    actions: list
        The action sequence of the path.
    """
    cum_reward = 0.0
    actions = []
    s = model.getInitialState()
    for i in range(model.maxSteps):
        # if verbose:
        # 	print("Step: ",i," of ", model.maxSteps)
        a = policy(p, s)
        actions.append(a)
        s, r = model.getNextState(s, a)
        time.sleep(sleeptime)
        cum_reward += r
        if model.isEndState(s):
            break
    if verbose:
        print("End at: ", i, " of ", model.maxSteps)
    return cum_reward, actions
