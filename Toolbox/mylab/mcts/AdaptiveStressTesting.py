import copy
import mylab.mcts.MDP as MDP
import mylab.mcts.MCTSdpw as MCTSdpw
import numpy as np
# import garage.misc.logger as logger
from dowel import logger, tabular
import pickle

class ASTParams:
    def __init__(self,max_steps,log_interval,log_tabular, log_dir=None):
        self.max_steps = max_steps
        self.log_interval = log_interval
        self.log_tabular = log_tabular
        self.log_dir = log_dir

class AdaptiveStressTest:
    def __init__(self,p,env,top_paths):
        self.params = p
        self.env = env
        self.sim_hash = hash(0)
        self.transition_model = self.transition_model()
        self.step_count = 0
        self._isterminal = False
        self._reward = 0.0
        self.action_seq = []
        self.trajectory_reward = 0.0
        self.top_paths = top_paths

    def reset_setp_count(self):
        self.step_count = 0
    def initialize(self):
        self._isterminal = False
        self._reward = 0.0
        self.action_seq = []
        self.trajectory_reward = 0.0
        return self.env.reset()
    def update(self,action):
        self.step_count += 1
        # print("step_count: ",self.step_count)
        obs, reward, done, info = self.env.step(action.get())
        # print("step: ",obs, reward, done)
        self._isterminal = done
        self._reward = reward
        self.action_seq.append(action)
        self.trajectory_reward += reward
        if done:
            self.top_paths.enqueue(self.action_seq,self.trajectory_reward,make_copy=True)
        if self.params.log_tabular:
            if self.step_count%self.params.log_interval == 0:
                logger.log(' ')
                # logger.record_tabular('StepNum',self.step_count)
                tabular.record('StepNum',self.step_count)
                record_num = 0
                if self.params.log_dir is not None:

                    if self.step_count == self.params.log_interval:
                        best_actions = []
                    else:
                        with open(self.params.log_dir + '/best_actions.p', 'rb') as f:
                            best_actions = pickle.load(f)

                    best_actions.append(np.array([x.get() for x in self.top_paths.pq[0][0]]))
                    with open(self.params.log_dir + '/best_actions.p', 'wb') as f:
                        pickle.dump(best_actions, f)

                for (topi, path) in enumerate(self.top_paths):
                    # logger.record_tabular('reward '+str(topi), path[0])
                    tabular.record('reward '+str(topi), path[1])
                    record_num += 1

                for topi_left in range(record_num, self.top_paths.N):
                    tabular.record('reward '+str(topi_left), 0)
                # logger.dump_tabular(with_prefix=False)
                logger.log(tabular)
                logger.dump_all(self.step_count)
                tabular.clear()
        return obs, reward, done, info
    def isterminal(self):
        return self._isterminal
    def get_reward(self):
        return self._reward
    def random_action(self):
        # action = self.env.action_space.sample()
        # print('action: ',action)
        # return ASTAction(action)
        return ASTAction(self.env.action_space.sample())
    def explore_action(self,s,tree):
        return ASTAction(self.env.action_space.sample())
    def transition_model(self):
        def get_initial_state():
            self.t_index = 1
            self.initialize()
            # s = ASTStateInit(ast.t_index, None, ASTAction(rsg=copy.deepcopy(ast.initial_rsg)))
            s = ASTStateInit(self.t_index, None, None)
            self.sim_hash = s.hash
            return s
        def get_next_state(s0,a0):
            assert self.sim_hash == s0.hash
            self.t_index += 1
            self.update(a0)
            s1 = ASTStateInit(self.t_index, s0, a0)
            self.sim_hash = s1.hash
            r = self.get_reward()
            return s1, r
        def isterminal(s):
            assert self.sim_hash == s.hash
            return self.isterminal()
        def go_to_state(target_state):
            s = get_initial_state()
            actions = get_action_sequence(target_state)
            # print("go to state with actions: ",actions)
            R = 0.0
            for a in actions:
                s,r = get_next_state(s, a)
                R += r
            assert s == target_state
            return R, actions
        return MDP.TransitionModel(get_initial_state, get_next_state, isterminal, self.params.max_steps, go_to_state)

class ASTState:
    def __init__(self,t_index,s_hash,parent,action):
        self.t_index = t_index
        self.s_hash = s_hash
        self.parent = parent
        self.action = action
    def __hash__(self):
        if self.parent is None:
            return hash((self.t_index,None,hash(self.action)))
        else:
            return hash((self.t_index,self.parent.s_hash,hash(self.action)))
    def __eq__(self,other):
        return hash(self) == hash(other)


def ASTStateInit(t_index,parent,action):
    obj = ASTState(t_index,0,parent,action)
    obj.hash = hash(obj)
    return obj

class ASTAction:
    def __init__(self,action):
        self.action = action
    def __hash__(self):
        return hash(tuple(self.action))
    def __eq__(self,other):
        return np.array_equal(self.action, other.action)
    def get(self):
        return self.action

def get_action_sequence(s):
    actions = []
    while not s.parent is None:
        actions.append(s.action)
        s = s.parent
    actions = list(reversed(actions))
    return actions