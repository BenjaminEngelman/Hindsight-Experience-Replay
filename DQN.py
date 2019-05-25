from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model
from ReplayMemory import ReplayMemory
import random
import numpy as np


SAVE_MODEL_PATH = "DQN-HER.h5"

def preprocess_input(input, size):
    return np.reshape(input, [1, size])

def argmax(l):
    """ Return the index of the maximum element of a list
    """
    return max(enumerate(l), key=lambda x: x[1])[0]

class NeuralNet:
    def __init__(self, state_size, action_space_size, goal_size, lr):        
        self.model = Sequential()
        self.model.add(Dense(256,  activation="relu", input_dim=state_size + goal_size))
        self.model.add(Dense(action_space_size, activation="linear"))
        adam = Adam(lr=lr)
        self.model.compile(optimizer=adam, loss="mse")
        self.model.summary()  
    
    def get_weights(self):
        return self.model.get_weights()
    
    def set_weights(self, weights):
        self.model.set_weights(weights)
    
    def predict(self, input):
        return self.model.predict(input)

    def fit(self, inputs, outputs, batch_size):
        return self.model.fit(inputs, outputs, epochs=1, batch_size=batch_size, verbose=0)

class DQN:
    def __init__(self,num_action, state_size, goal_size, max_eps=1, min_eps=0.1, eps_decay=0.975, gamma=0.98, lr=0.001, batch_size=128, buffer_size=1000000, init_nn=None):
        
        # Init Hyperparameters       
        self.epsilon = max_eps
        self.min_eps = min_eps
        self.eps_decay = eps_decay
        self.gamma = gamma
        self.num_action = num_action
        self.batch_size = batch_size
        self.epsilon = max_eps
        self.memory = ReplayMemory(buffer_size)

        self.state_size = state_size
        self.goal_size = goal_size
        
        # Initialize neural nets
        self.policy_net = NeuralNet(state_size, num_action, goal_size, lr)
        if init_nn is not None:
            self.policy_net = init_nn

        self.target_net = NeuralNet(state_size, num_action, goal_size, lr)
        self.target_net.set_weights(self.policy_net.get_weights())



        
    def get_action(self, state, goal, greedy=False):
        '''Return an action according to the eps-greedy policy
        '''
        if random.random() < self.epsilon and not greedy:
            action = random.randrange(self.num_action)
        else:
            input = np.concatenate((state, goal), axis=0)
            input = preprocess_input(input, self.state_size * 2)
            action = argmax(self.policy_net.predict(input)[0]) 
        return action
    
    def remember(self, experiences):
        '''Store a list of [state, action, reward, next state, done, goal] experiences in the memory
        '''
        for exp in experiences:
            self.memory.push(exp)
    
    def replay(self, optimization_step):
        '''Get a batch of exeriences from the memory and then train the policy
        network on those experiences
        '''
        if self.memory.size() > self.batch_size:
            for opt_step in range(optimization_step):
                batch = self.memory.sample(self.batch_size)
                inputs, targets = [], [] # Will be used to batch fit the policy net (it's way faster)
                
                for state, action, reward, next_state, done, goal in batch:
                    target = reward
                    if not done:
                        input_next_state = np.concatenate((next_state, goal))
                        input_next_state = preprocess_input(input_next_state, self.state_size * 2)
                        next_qvals = self.target_net.predict(input_next_state)[0]

                        best_action = np.argmax(self.policy_net.predict(input_next_state))
                        target = reward + self.gamma * next_qvals[best_action]
                    
                    input_state = np.concatenate((state, goal), axis=0)
                    input_state = preprocess_input(input_state, self.state_size * 2)
                    q_vals = self.policy_net.predict(input_state)[0]
                    q_vals[action] = target
                    inputs.append(input_state[0])
                    targets.append(q_vals)
                
                self.policy_net.fit(np.array(inputs), np.array(targets), self.batch_size)

            
    def next_episode(self, i):
        '''Updates the exploration rate and copy the weights of the policy net
        into the target network
        '''
        # Decrease exlporation        
        if self.epsilon > self.min_eps:
            self.epsilon *= self.eps_decay
            self.epsilon = max(self.min_eps, self.epsilon)
        
        # Update target net
        self.target_net.set_weights(self.policy_net.get_weights())
        
        # Save model
        self.policy_net.model.save(SAVE_MODEL_PATH)