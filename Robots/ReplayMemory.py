from collections import deque
import random
import numpy as np
from copy import deepcopy


class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(deepcopy(transition))

    def sample(self, batch_size):
        batch_idxs = np.random.randint(self.size() - 2, size=batch_size)
        batch = [deepcopy(self.memory[batch_idx]) for batch_idx in batch_idxs]

        s_batch = np.array([i[0] for i in batch])
        a_batch = np.array([i[1] for i in batch])
        r_batch = np.array([i[2] for i in batch])
        new_s_batch = np.array([i[3] for i in batch])
        ag_batch = np.array([i[4] for i in batch])
        g_batch = np.array([i[5] for i in batch])

        return s_batch, a_batch, r_batch, new_s_batch, ag_batch, g_batch
        

    def size(self):
        return len(self.memory)