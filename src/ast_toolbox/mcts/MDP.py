import time


def identity(*args):
    if len(args) == 1:
        return args[0]
    return args


class TransitionModel:
    def __init__(self, getInitialState, getNextState, isEndState, maxSteps, goToState):
        self.getInitialState = getInitialState
        self.getNextState = getNextState
        self.isEndState = isEndState
        self.maxSteps = maxSteps
        self.goToState = goToState


def simulate(model, p, policy, verbose=False, sleeptime=0.0):
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
