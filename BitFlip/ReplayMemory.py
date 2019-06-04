from collections import deque
from sumtree import SumTree
import random
import numpy as np
from copy import deepcopy

class ReplayMemory:

    def __init__(self, capacity, with_priorities=False):
        self.capacity = capacity
        self.with_priorities = with_priorities
        if with_priorities:
            self.alpha = 0.5
            self.epsilon = 0.01
            self.memory = SumTree(capacity)
        else:
            self.memory = deque(maxlen=capacity)

    def push(self, transition):
        
        if self.with_priorities:
            td_error = transition[-1]
            priority = self.priority(td_error)
            self.memory.add(priority, transition)
        else:
            self.memory.append(transition)

    
    def priority(self, error):
        return (error + self.epsilon) ** self.alpha

    def sample(self, batch_size):
        batch = []
        if self.with_priorities:
            T = self.memory.total() // batch_size
            for i in range(batch_size):
                a, b = T * i, T * (i + 1)
                s = random.uniform(a, b)
                idx, _, data = self.memory.get(s)
                batch.append(deepcopy([*data, idx]))
        else:
            batch_idxs = np.random.randint(self.size() - 2, size=batch_size)
            for batch_idx in batch_idxs:
                sample = deepcopy(self.memory[batch_idx])
                sample.append(None)
                batch.append(sample)
            # batch = [self.memory[batch_idx] for batch_idx in batch_idxs]

        return batch
    
    def update(self, idx, new_error):
        """ Update priority for idx (PER)
        """
        self.memory.update(idx, self.priority(new_error))

    def size(self):
        return len(self.memory)

        