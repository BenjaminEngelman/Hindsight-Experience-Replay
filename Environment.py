import numpy as np

class BitFlippingEnv():
    def __init__(self, n):
        self.n = n
    
    def reset(self):
        self.goal = np.random.randint(2, size=(self.n))
        self.state =  np.random.randint(2, size=(self.n))

        return self.state, self.goal
    
    def step(self, action):
        # An action is a position between 0 and n - 1
        assert(action >= 0 and action < self.n)
        self.state[action] = 1 if self.state[action] == 0 else 0 # flip the bit

        done = np.array_equal(self.state, self.goal)
        reward = 1 if done else -1

        return np.copy(self.state), reward, done
    
    def render(self):
        print(self.state.tolist())