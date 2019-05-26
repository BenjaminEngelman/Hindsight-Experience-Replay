from collections import deque
import random

class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        sample_size = min(batch_size, len(self.memory))
        return random.sample(self.memory, sample_size)

    def size(self):
        return len(self.memory)