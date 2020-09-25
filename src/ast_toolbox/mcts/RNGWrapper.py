
import gym.utils.seeding as seeding
import numpy as np


class RSG:
    """The random seed generator for AST using random seeds.

    Parameters
    ----------
    state_length : int, optional
        The length of the RSG state.
    seed : int, optional
        The initial seed to generate the initial state.
    """

    def __init__(self, state_length=1, seed=0):
        self.state = seed_to_state_itr(state_length, seed)

    def __eq__(self, other):
        """The redefined equal function.

        Returns
        ----------
        is_equal : bool
            Whether the two RSG are equal.
        """
        return np.array_equal(self.state, other.state)

    def __hash__(self):
        """The redefined hashing function.

        Returns
        ----------
        hash : int
            The hashing result.
        """
        return hash(tuple(self.state))

    def length(self):
        """Return the length of the RSG state.

        Returns
        ----------
        length : int
            The length of the RSG state.
        """
        return len(self.state)

    def next(self):
        """Step the RSG state.
        """
        self.state = np.array(list(map(seeding.hash_seed, self.state)), dtype=np.uint32)

    def set_from_seed(self, length, seed):
        """Set the RSG state using the seed.

        Parameters
        ----------
        length : int
            The length of the RSG state.
        seed : int
            The seed to generate the state.
        """
        self.state = seed_to_state_itr(length, seed)


def seed_to_state_itr(state_length, seed):
    """Generate the RSG state using the seed.

    Parameters
    ----------
    state_length : int
        The length of the RSG state.
    seed : int
        The seed to generate the state.

    Returns
    ----------
    state : :py:class:`numpy.ndarry`
        The generated state.
    """
    state = []
    seedi = seed
    for i in range(state_length):
        # print(seedi)
        state.append(seedi)
        seedi = seeding.hash_seed(seedi)
    return np.array(state, dtype=np.uint32)
