import time

import numpy as np


class DPWParams:
    """Structure that stores the parameters for the MCTS with DPW.

    Parameters
    ----------
    d : int
        The maximum searching depth.
    gamma : float
        The discount factor.
    ec : float
        The weight for the exploration bonus.
    n : int
        The mximum number of iterations.
    k : float
        The constraint parameter used in DPW: |N(s,a)|<=kN(s)^alpha.
    alpha : float
        The constraint parameter used in DPW: |N(s,a)|<=kN(s)^alpha.
    clear_nodes : bool
        Whether to clear redundant nodes in tree.
        Set it to True for saving memoray. Set it to False to better tree plotting.
    """

    def __init__(self, d, gamma, ec, n, k, alpha, clear_nodes):
        self.d = d  # search depth
        self.gamma = gamma  # discount factor
        self.ec = ec  # exploration constant
        self.n = n  # number of iterations
        self.k = k  # dpw parameters
        self.alpha = alpha  # dpw parameters
        self.clear_nodes = clear_nodes


class DPWModel:
    """The model used in the tree search.

    Parameters
    ----------
    model : :py:class:`ast_toolbox.mcts.MDP.TransitionModel`
        The transition model.
    getAction : function
        getAction(s, tree) returns the action used in rollout.
    getNextAction : function
        getNextAction(s, tree) returns the action used in exploration.
    """

    def __init__(self, model, getAction, getNextAction):
        self.model = model
        self.getAction = getAction  # expert action used in rollout
        self.getNextAction = getNextAction  # exploration strategy


class StateActionStateNode:
    """The structure storing the transition state-action-state.
    """

    def __init__(self):
        self.n = 0  # UInt64
        self.r = 0.0  # Float64


class StateActionNode:
    """The structure representing the state-action node.
    """

    def __init__(self):
        self.s = {}  # Dict{State,StateActionStateNode}
        self.n = 0  # UInt64
        self.q = 0.0  # Float64


class StateNode:
    """The structure representing the state node.
    """

    def __init__(self):
        self.a = {}  # Dict{Action,StateActionNode}
        self.n = 0  # UInt64


class DPWTree:
    """The structure storing the seaching tree.
    """

    def __init__(self, p, f):
        self.s_tree = {}  # Dict{State,StateNode}
        self.p = p  # DPWParams
        self.f = f  # DPWModel


def saveBackwardState(old_s_tree, new_s_tree, s_current):
    """Saving the s_current as well as all its predecessors in the old_s_tree into the new_s_tree.

    Parameters
    ----------
    old_s_tree : dict
        The old tree.
    new_s_tree : dict
        The new tree.
    s_current : :py:class:`ast_toolbox.mcts.AdaptiveStressTesting.ASTState`
        The current state.

    Returns
    ----------
    new_s_tree : dict
        The new tree.
    """
    if not (s_current in old_s_tree):
        return new_s_tree
    s = s_current
    while s is not None:
        new_s_tree[s] = old_s_tree[s]
        s = s.parent
    return new_s_tree


def saveForwardState(old_s_tree, new_s_tree, s):
    """Saving the s_current as well as all its successors in the old_s_tree into the new_s_tree.

    Parameters
    ----------
    old_s_tree : dict
        The old tree.
    new_s_tree : dict
        The new tree.
    s_current : :py:class:`ast_toolbox.mcts.AdaptiveStressTesting.ASTState`
        The current state.

    Returns
    ----------
    new_s_tree : dict
        The new tree.
    """
    if not (s in old_s_tree):
        return new_s_tree
    new_s_tree[s] = old_s_tree[s]
    for sa in old_s_tree[s].a.values():
        for s1 in sa.s.keys():
            saveForwardState(old_s_tree, new_s_tree, s1)
    return new_s_tree


def saveState(old_s_tree, s):
    """Saving the s_current as well as all its predecessors and successors in the old_s_tree into the new_s_tree.

    Parameters
    ----------
    old_s_tree : dict
        The old tree.
    s : :py:class:`ast_toolbox.mcts.AdaptiveStressTesting.ASTState`
        The current state.

    Returns
    ----------
    new_s_tree : dict
        The new tree.
    """
    new_s_tree = {}
    saveBackwardState(old_s_tree, new_s_tree, s)
    saveForwardState(old_s_tree, new_s_tree, s)
    return new_s_tree


def selectAction(tree, s, verbose=False):
    """Run MCTS to select one action for the state s

    Parameters
    ----------
    tree : :py:class:`ast_toolbox.mcts.MCTSdpw.DPWTree`
        The seach tree.
    s : :py:class:`ast_toolbox.mcts.AdaptiveStressTesting.ASTState`
        The current state.
    verbose : bool, optional
        Where to log the seaching information.

    Returns
    ----------
    action : `ast_toolbox.mcts.AdaptiveStressTesting.ASTAction`
        The selected AST action.
    """
    if tree.p.clear_nodes:
        new_dict = saveState(tree.s_tree, s)
        tree.s_tree.clear()
        tree.s_tree = new_dict

    depth = tree.p.d
    time.time() * 1e6
    for i in range(tree.p.n):
        R, actions = tree.f.model.goToState(s)
        R += simulate(tree, s, depth, verbose=verbose)

    tree.f.model.goToState(s)
    state_node = tree.s_tree[s]
    explored_actions = list(state_node.a.keys())
    nA = len(explored_actions)
    Q = np.zeros(nA)
    for i in range(nA):
        Q[i] = state_node.a[explored_actions[i]].q
    assert len(Q) != 0
    i = np.argmax(Q)
    return explored_actions[i]


def simulate(tree, s, depth, verbose=False):
    """Single run of the forward MCTS search.

    Parameters
    ----------
    tree : :py:class:`ast_toolbox.mcts.MCTSdpw.DPWTree`
        The seach tree.
    s : :py:class:`ast_toolbox.mcts.AdaptiveStressTesting.ASTState`
        The current state.
    depth : int
        The maximum search depth
    verbose : bool, optional
        Where to log the seaching information.

    Returns
    ----------
    q : float
        The estimated return.
    """
    if (depth == 0) | tree.f.model.isEndState(s):
        return 0.0

    if not (s in tree.s_tree):
        tree.s_tree[s] = StateNode()
        return rollout(tree, s, depth)

    tree.s_tree[s].n += 1
    if len(tree.s_tree[s].a) < tree.p.k * tree.s_tree[s].n**tree.p.alpha:
        # explore new action
        a = tree.f.getNextAction(s, tree.s_tree)
        if not (a in tree.s_tree[s].a):
            tree.s_tree[s].a[a] = StateActionNode()
    else:
        # sample explored actions
        state_node = tree.s_tree[s]
        explored_actions = list(state_node.a.keys())
        nA = len(explored_actions)
        UCT = np.zeros(nA)
        nS = state_node.n
        assert nS > 0
        for i in range(nA):
            state_action_node = state_node.a[explored_actions[i]]
            assert state_action_node.n > 0
            UCT[i] = state_action_node.q + tree.p.ec * np.sqrt(np.log(nS) / float(state_action_node.n))
        a = explored_actions[np.argmax(UCT)]

    tree.s_tree[s].a[a].q

    sp, r = tree.f.model.getNextState(s, a)
    if not (sp in tree.s_tree[s].a[a].s):
        tree.s_tree[s].a[a].s[sp] = StateActionStateNode()
        tree.s_tree[s].a[a].s[sp].r = r
        tree.s_tree[s].a[a].s[sp].n = 1
    else:
        tree.s_tree[s].a[a].s[sp].n += 1

    q = r + tree.p.gamma * simulate(tree, sp, depth - 1)
    state_action_node = tree.s_tree[s].a[a]
    state_action_node.n += 1
    state_action_node.q += (q - state_action_node.q) / float(state_action_node.n)
    tree.s_tree[s].a[a] = state_action_node

    return q


def rollout(tree, s, depth):
    """Rollout from the current state s.

    Parameters
    ----------
    tree : :py:class:`ast_toolbox.mcts.MCTSdpw.DPWTree`
        The seach tree.
    s : :py:class:`ast_toolbox.mcts.AdaptiveStressTesting.ASTState`
        The current state.
    depth : int
        The maximum search depth

    Returns
    ----------
    q : float
        The estimated return.
    """
    if (depth == 0) | tree.f.model.isEndState(s):
        return 0.0
    else:
        a = tree.f.getAction(s, tree.s_tree)
        sp, r = tree.f.model.getNextState(s, a)
        qval = (r + rollout(tree, sp, depth - 1))
        return qval
