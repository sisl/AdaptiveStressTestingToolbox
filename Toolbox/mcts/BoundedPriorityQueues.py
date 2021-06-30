import copy

import numpy as np
from depq import DEPQ


class BoundedPriorityQueue:
    """The bounded priority Queue.

    Parameters
    ----------
    N : int
        Size of the queue.
    """

    def __init__(self, N):
        self.pq = DEPQ(iterable=None, maxlen=N)
        self.N = N

    def enqueue(self, k, v, make_copy=False):
        """Storing k into the queue based on the priority value v.

        Parameters
        ----------
        k :
            The object to be stored.
        v : float
            The priority value.
        make_copy : bool, optional
            Whether to make a copy of the k.
        """
        if isinstance(k, np.ndarray):
            for pair in self.pq.data:
                if np.array_equal(k, pair[0]):
                    return
        elif k in [pair[0] for pair in self.pq]:
            return
        while v in [pair[1] for pair in self.pq]:
            v += 1e-4
        if make_copy:
            ck = copy.deepcopy(k)
            self.pq.insert(ck, v)
        else:
            self.pq.insert(k, v)

    def length(self):
        """Return the current size of the queue.

        Returns
        ----------
        length : int
            The current size of the queue.
        """
        return self.pq.size()

    def empty(self):
        """Clear the queue.
        """
        self.pq.clear()

    def isempty(self):
        """Check whether the queue is empty.

        Returns
        ----------
        is_empty : bool
            Whether the queue is empty.
        """
        return self.pq.is_empty()

    def haskey(self, k):
        """Check whether k in in the queue.

        Returns
        ----------
        has_key : bool
            Whether k in in the queue.
        """
        return k in [pair[0] for pair in self.pq]

    def __iter__(self):
        """The redefined iteration function.

        Returns
        ----------
        BPQ_Iterator : generator
            The BPQ iterator.
        """
        # return start(self)
        kvs = list(reversed(sorted(self.pq, key=lambda x: x[1])))
        return (kv for kv in kvs)
