import copy

import numpy as np
from depq import DEPQ


class BoundedPriorityQueue:
    def __init__(self, N):
        self.pq = DEPQ(iterable=None, maxlen=N)
        self.N = N

    def enqueue(self, k, v, make_copy=False):
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
        return self.pq.size()

    def empty(self):
        self.pq.clear()

    def isempty(self):
        return self.pq.is_empty()

    def haskey(self, k):
        return k in [pair[0] for pair in self.pq]

    def __iter__(self):
        return start(self)


class BPQIterator:
    def __init__(self, sorted_pairs, index):
        self.sorted_pairs = sorted_pairs
        self.index = index

    def __next__(self):
        if self.index > len(self.sorted_pairs) - 1:
            raise StopIteration
        else:
            item = self.sorted_pairs[self.index]
            self.index += 1
            return item


def start(q):
    kvs = list(reversed(sorted(q.pq, key=lambda x: x[1])))
    return BPQIterator(kvs, 0)
