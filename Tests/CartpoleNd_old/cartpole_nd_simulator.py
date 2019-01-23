from mylab.simulators.nn_sut_simulator import NNSUTSimulator
from CartpoleNd.cartpole_nd import CartPoleNdEnv


class CartpoleNdSimulator(NNSUTSimulator):
    """
    neural network system under test simulator
    """

    def __init__(self,
    			nd,
    			use_seed=False,
                **kwargs):

        #initialize the base Simulator
        super().__init__(**kwargs,env=CartPoleNdEnv(nd=nd,use_seed=use_seed))
