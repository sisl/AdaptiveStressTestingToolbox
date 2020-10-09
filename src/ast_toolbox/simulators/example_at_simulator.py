import os
import sys
from pprint import pprint  # pretty print (for lists)

import matlab.engine
import numpy as np  # Used for math
import torch

from ast_toolbox.simulators import ASTSimulator  # import base Simulator class

sys.path.insert(0, '../stlcg/src')
import stlcg  # noqa

# globalize MATLAB engine for DRL.
ENG = None


# Define the class: Automatic Transmission Simulink Model (falsification baseline)
class ExampleATSimulator(ASTSimulator):
    """
    Class template for a non-interactive simulator.
    """
    # Accept parameters for defining the behavior of the system under test[SUT]

    def __init__(self, **kwargs):
        global ENG
        if ENG is None:
            model_name = "Autotrans_shift"
            print("Loading Simulink model: " + model_name)
            ENG = matlab.engine.start_matlab()
            ENG.load_system(os.path.dirname(os.path.abspath(__file__)) + "/../models/" + model_name + ".mdl")
            print("Done!")

        # Constant hyper-params -- set by user
        self.use_01_formulation = False  # Use the [0, 1] representation of the action space. Note, duplicate in "ExampleATSpaces".
        self.use_stl_robustness = True  # Use STL robustness instead of direct miss distance. (testing)
        self.use_end_action_interpolation = False  # append action out of time range to allow linear interpolation (from MATLAB docs)
        self.use_time_rejection = False  # reject actions if the time component is in the past
        self.augment_time_left = False  # NOTE: Change ExampleATSpace max_time = 1
        self.verbose = False

        self.model_name = "Autotrans_shift"
        self.results_name = "results"

        # If MATLAB engine is to be local, uncomment these.
        # print("Loading Simulink model: " + self.model_name)
        # self.eng = matlab.engine.start_matlab()
        # self.eng.load_system(os.path.dirname(os.path.abspath(__file__)) + "/../models/" + self.model_name + ".mdl")
        # print("Done!")

        self._miss_distance = np.inf

        self._max_sim_time = 30
        self._benchmark = os.environ.get('AT')  # "AT1", "AT2", "AT3", "AT4"
        if self._benchmark is None:
            self._benchmark = "AT1"  # default to AT1 if not specified by the `run_algo.py` script
        print("Benchmark: ", self._benchmark)

        self._at1_max_time = 20  # 30 seconds in FalStar paper, 20 in ARCH-COMP
        self._at1_speed_thresh = 120

        self._at2_max_time = 10  # 10 in ARCH-COMP (FalStar paper has different AT2 altogether)
        self._at2_rpm_thresh = 4750

        self._actionset = self.get_first_action()
        self._state = None
        self._times = np.array([])
        self._speeds = np.array([])
        self._rpms = np.array([])
        self._gears = np.array([])
        self._reject_time_inconsistency = False

        # temporal logic specifications
        self._formula_at1 = stlcg.Always(subformula=stlcg.LessThan(lhs='s', val=self._at1_speed_thresh), interval=[0, self._at1_max_time])
        self._formula_at2 = stlcg.Always(subformula=stlcg.LessThan(lhs='w', val=self._at2_rpm_thresh), interval=[0, self._at2_max_time])

        # initialize the base Simulator
        super().__init__(**kwargs)

    def get_first_action(self):
        if self.use_end_action_interpolation:
            # NOTE: Based on the MATLAB documentation, this is to ensure linear interpolation occurs
            # (https://www.mathworks.com/help/simulink/slref/modeling-an-automatic-transmission-controller.html)
            return np.array([100, 0, 0])
        else:
            return np.array([0])

    def simulate(self, actions, s_0):
        return

    def closed_loop_step(self, action):
        """
        Handle anything that needs to take place at each step, such as a simulation update or write to file
        Input
        -----
        action : action taken on the turn
        Outputs
        -------
        (terminal_index)
        terminal_index : The index of the action that resulted in a state in the goal set E. If no state is found
                        terminal_index should be returned as -1.
        """
        self._reject_time_inconsistency = False

        local_action = action.copy()

        # append the action to the list of all actions
        if len(self._actionset) == 0:
            if self.augment_time_left:
                local_action[0] = local_action[0] * self._max_sim_time  # augment [0,1] to "time left"
            self._actionset = np.array([local_action])
        else:
            if self.augment_time_left:  # NOTE: Make sure ExampleATSpace has max_time of 1 (not 30).
                # Augment [0, 1] "time left"
                local_action[0] = self._actionset[-1][0] + (self._max_sim_time - self._actionset[-1][0]) * local_action[0]

            # reject if time is smaller than latest action
            self._reject_time_inconsistency = local_action[0] < self._actionset[-1][0]
            if not self.use_time_rejection or not self._reject_time_inconsistency:
                self._actionset = np.vstack((self._actionset, local_action))

        if not self.use_time_rejection or not self._reject_time_inconsistency:
            (self._speeds, self._rpms, self._gears, self._times) = self.evaluate(self._actionset)

        # grab simulation state, if interactive
        self.observe()
        self.observation = self._miss_distance

        if self.is_goal():
            t0 = 0
            t1 = np.inf
            metric = None
            if self._benchmark == "AT1":
                metric = self._speeds
                t1 = self._at1_max_time
            elif self._benchmark == "AT2":
                metric = self._rpms
                t1 = self._at2_max_time
            print("————————======== FAILURE FOUND! ========———————— [", t0, ", ", t1, "] ", self.max_in_time(t0, t1, metric, self._times))

        return self.observation_return()

    def max_in_time(self, t0, t1, Y, T):
        return max(y if t0 <= t <= t1 else -np.inf for (t, y) in zip(T, Y))

    # Organize Simulink input call.
    def collect_inputs(self, x):
        global ENG

        # Collect Simulink inputs (note use of ' instead of ")
        stop_time = ('StopTime', str(self._max_sim_time))
        load_external_input = ('LoadExternalInput', 'on')
        external_input = ('ExternalInput', ENG.mat2str(matlab.double(x)))  # [xi.tolist() for xi in x]
        save_time = ('SaveTime', 'on')
        time_save_name = ('TimeSaveName', 'tout')
        save_output = ('SaveOutput', 'on')
        output_save_name = ('OutputSaveName', 'yout')
        save_format = ('SaveFormat', 'Array')
        solver_mode = ('SolverType', 'variable-step')  # Needed to force integer output time.
        output_options = ('OutputOption', 'SpecifiedOutputTimes')  # Force integer time.
        zero_cross_alg = ('ZeroCrossAlgorithm', 'Adaptive')
        output_time_resolution = ('OutputTimes', ENG.mat2str(matlab.double(range(self._max_sim_time+1))))

        inputs = [stop_time, load_external_input,
                  external_input, save_time,
                  time_save_name, save_output,
                  output_save_name, save_format,
                  solver_mode, output_options,
                  zero_cross_alg, output_time_resolution]

        # Formatted input arguments: quoted and comma separated.
        return ", ".join("'{0}'".format(arg) for arg in [p for param in inputs for p in param])

    def collect_outputs(self):
        global ENG
        Y = ENG.eval(self.results_name + ".yout'")
        speeds = Y[0]
        rpms = Y[1]
        gears = Y[2]
        times = ENG.eval(self.results_name + ".tout'")[0]
        assert(len(times) == self._max_sim_time+1)  # Ensure integer time steps
        if self.verbose:
            if self._benchmark == "AT1" or self._benchmark == "AT3":
                print("Max. speed = ", max(speeds), "\t|\t Mean speed = ", np.mean(speeds))
                print("\n—————————————————————")
            elif self._benchmark == "AT2":
                print("Max. RPMs = ", max(rpms), "\t|\t Mean RPMs = ", np.mean(rpms))
                print("\n—————————————————————")
        return (speeds, rpms, gears, times)

    # Evaluate Simulink model.
    def evaluate(self, x):
        global ENG
        x = np.array(x)

        if self.use_01_formulation:
            # x = np.concatenate(x)
            x = x*[30, 100, 325]  # convert from [0-1] to proper ranges

        # if not self.use_time_rejection:
        x = x[x[:, 0].argsort()]  # sort by time column
        x = x.tolist()

        if self.verbose:
            pprint(x)

        input_arguments = self.collect_inputs(x)
        ENG.evalc(self.results_name + " = sim('" + self.model_name + "'," + input_arguments + ");", nargout=0)
        return self.collect_outputs()

    def reset(self, s_0):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        self.initial_conditions = self.get_first_action()  # inherited
        self._first_step = True
        self._actionset = np.array([])
        self._is_terminal = False  # inherited
        self._path_length = 0  # inherited
        self.observation = None
        self._reject_time_inconsistency = False

        return self.observation_return()

    def get_reward_info(self):
        """
        returns any info needed by the reward function to calculate the current reward
        """
        return {"d": self._miss_distance,
                "is_goal": self.is_goal(),
                "is_terminal": self._is_terminal}

    # Temporal logic.
    def implies(self, p, q):
        return q if p else True

    def always(self, t0, t1, condition, T, Y):
        return all(condition(y) if t0 <= t <= t1 else 1 for (t, y) in zip(T, Y))

    def robustness(self, t0, t1, miss, T, Y):
        return [miss(y) if t0 <= t <= t1 else np.inf for (t, y) in zip(T, Y)]

    def eventually(self, t0, t1, condition, T, Y):
        return any(condition(y) if t0 <= t <= t1 else 0 for (t, y) in zip(T, Y))

    def AT1(self, T, speeds):
        if self.use_stl_robustness:
            if len(speeds) == 0:
                return True  # bypass.
            else:
                signal = torch.tensor(speeds).unsqueeze(0).unsqueeze(-1).flip(1)
                return self._formula_at1.eval(signal).squeeze().item()
        else:
            return self.always(0, self._at1_max_time, lambda s: s < self._at1_speed_thresh, T, speeds)

    def AT2(self, T, rpms):
        if self.use_stl_robustness:
            if len(rpms) == 0:
                return True  # bypass.
            else:
                signal = torch.tensor(rpms).unsqueeze(0).unsqueeze(-1).flip(1)
                return self._formula_at2.eval(signal).squeeze().item()
        else:
            return self.always(0, self._at2_max_time, lambda w: w < self._at2_rpm_thresh, T, rpms)

    def is_goal(self):
        """
        returns whether the current state is in the goal set
        :return: boolean, true if current state is in goal set.
        """
        if self._benchmark == "AT1":
            return not self.AT1(self._times, self._speeds)
        elif self._benchmark == "AT2":
            return not self.AT2(self._times, self._rpms)
        else:
            error("No benchmark named ", self._benchmark)  # noqa

    def observe(self):
        if self._reject_time_inconsistency:
            self._miss_distance = self._at1_speed_thresh
        else:
            if self.use_stl_robustness:
                if self._benchmark == "AT1":
                    signal = torch.tensor(self._speeds).unsqueeze(0).unsqueeze(-1).flip(1)
                    self._miss_distance = self._formula_at1.robustness(signal).squeeze().item()
                elif self._benchmark == "AT2":
                    signal = torch.tensor(self._rpms).unsqueeze(0).unsqueeze(-1).flip(1)
                    self._miss_distance = self._formula_at2.robustness(signal).squeeze().item()
                else:
                    error("observe(), no benchmark named ", self._benchmark)  # noqa
            else:
                if self._benchmark == "AT1":
                    # NOTE: (thresh - s) so negative miss is past event.
                    self._miss_distance = min(self.robustness(0, self._at1_max_time,
                                                              lambda s: self._at1_speed_thresh - s,
                                                              self._times,
                                                              self._speeds))
                elif self._benchmark == "AT2":
                    # NOTE: (thresh - s) so negative miss is past event.
                    self._miss_distance = min(self.robustness(0, self._at2_max_time,
                                                              lambda w: self._at2_rpm_thresh - w,
                                                              self._times,
                                                              self._rpms))
                else:
                    error("observe(), no benchmark named ", self._benchmark)  # noqa

    def restore_state(self, in_simulator_state):
        simulator_state = in_simulator_state.copy()
        return simulator_state

    def _get_obs(self):
        return self._miss_distance

    def clone_state(self):
        return np.array([])
