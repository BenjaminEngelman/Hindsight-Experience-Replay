import sys
import numpy as np
import time
from tqdm import tqdm
from actor import Actor
from critic import Critic
from stats import gather_stats
from networks import tfSummary, OrnsteinUhlenbeckProcess
from memory_buffer import MemoryBuffer


NUM_EPOCHS = 200
NUM_CYCLES = 50
NUM_EPISODES = 16
OPTIMIZATION_STEPS = 40

class DDPG:
    """ Deep Deterministic Policy Gradient (DDPG) Helper Class
    """

    def __init__(self, act_dim, env_dim, act_range, k, buffer_size = 20000, gamma = 0.99, lr = 0.00005, tau = 0.001):
        """ Initialization
        """
        # Environment and A2C parameters
        self.act_dim = act_dim
        self.act_range = act_range
        self.env_dim = env_dim
        self.gamma = gamma
        self.lr = lr
        # Create actor and critic networks
        self.actor = Actor(self.env_dim, act_dim, act_range, 0.1 * lr, tau)
        self.critic = Critic(self.env_dim, act_dim, lr, tau)
        self.buffer = MemoryBuffer(buffer_size)

    def policy_action(self, s, g):
        """ Use the actor to predict value
        """

        input = np.concatenate((s,g), axis=0)
        return self.actor.predict(input)[0]

    def bellman(self, rewards, q_values, dones):
        """ Use the Bellman Equation to compute the critic target
        """
        critic_target = np.asarray(q_values)
        for i in range(q_values.shape[0]):
            if dones[i]:
                critic_target[i] = rewards[i]
            else:
                critic_target[i] = rewards[i] + self.gamma * q_values[i]
        return critic_target

    def memorize(self, experiences):
        """ Store experience in memory buffer
        """
        for exp in experiences:
            state, action, reward, done, new_state, ag, goal = exp
            self.buffer.memorize(state, action, reward, done, new_state, ag, goal)

    def sample_batch(self, batch_size):
        return self.buffer.sample_batch(batch_size)

    def update_models(self, states, actions, critic_target):
        """ Update actor and critic networks from sampled experience
        """
        # Train critic
        self.critic.train_on_batch(states, actions, critic_target)
        # Q-Value Gradients under Current Policy
        actions = self.actor.model.predict(states)
        grads = self.critic.gradients(states, actions)
        # Train actor
        self.actor.train(states, actions, np.array(grads).reshape((-1, self.act_dim)))
        # Transfer weights to target networks at rate Tau
        self.actor.transfer_weights()
        self.critic.transfer_weights()

    def train(self, env, args):
        success_rates = []
        for ep_num in range(NUM_EPOCHS):
            start = time.time()
            successes = 0
            for _ in range(NUM_CYCLES):
                for _ in range(NUM_EPISODES):
                    # Reset episode 
                    ep_time = 0
                    observation = env.reset()
                    old_state = observation['observation']
                    achieved_goal = observation['achieved_goal']
                    goal = observation['desired_goal']
                    noise = OrnsteinUhlenbeckProcess(size=self.act_dim)    
                    episode_exp = []
                    episode_exp_her = []
                    done = False

                    while not done:
                        if args['render']: env.render()
                        # Actor picks an action (following the deterministic policy)
                        action = self.policy_action(old_state, goal)
                        # Clip continuous values to be valid w.r.t. environment
                        action = np.clip(action+noise.generate(ep_time), -self.act_range, self.act_range)
                        # Retrieve new state, reward, and whether the state is terminal
                        obs, reward, done, _ = env.step(action)
                        new_state = obs['observation']
                        achieved_goal = obs['achieved_goal']
                        # Add outputs to memory buffer
                        episode_exp.append([old_state, action, reward, done, new_state, achieved_goal, goal])
                        old_state = new_state
                        ep_time += 1
                    
                    successes += achieved_goal == goal 

                    if args["HER_strat"] == "final":
                        experience = episode_exp[-1]
                        experience[3] = True # set done = true
                        experience[-1] = experience[-2] # set g' to achieved goal
                        experience[2] = 1 # set reward of success
                        episode_exp_her.append(experience)
                    
                    elif args["HER_strat"] in ["future", "eisode"]:
                        # For each transition of the episode trajectory
                        for t in range(len(episode_exp)):
                            # Add K random states which come from the same episode as the transition
                            for _ in range(args["HER_k"]):
                                if args["HER_strat"] == "future":
                                    # Select a future exp from the same episod
                                    selected = np.random.randint(t, len(episode_exp)) 
                                elif args["HER_strat"] == "episode":
                                    # Select an exp from the same episode
                                    selected = np.random.randint(0, len(episode_exp))  
                                _, _, _, _, _, g, _ = episode_exp[selected] # g = achieved goal of selected
                                s, a, _, _, ns, ag, _ = episode_exp[t]
                                d = np.array_equal(achieved_goal, g)
                                r = 1 if d else -1
                                episode_exp_her.append([s, a, r, d, ns, ag, g])
                    
                    self.memorize(episode_exp)
                    self.memorize(episode_exp_her)
                    
                for _ in range(OPTIMIZATION_STEPS):
                    # Sample experience from buffer
                    states, actions, rewards, dones, new_states, _, goals,_ = self.sample_batch(args["batch_size"])
                    inputs = np.concatenate((new_states, goals), axis=1)
                    q_values = self.critic.target_predict([inputs, self.actor.target_predict(inputs)])
                    # Compute critic target
                    critic_target = self.bellman(rewards, q_values, dones)
                    # Train both networks on sampled batch, update target networks
                    inputs = np.concatenate((states, goals), axis=1)

                    self.update_models(inputs, actions, critic_target)
                    
            success_rates.append(successes / (NUM_CYCLES * NUM_EPISODES))
            print("Epoch:", ep_num+1, " -- success rate:", success_rates[-1], " -- duration:", time.time() - start)



        return success_rates

    def save_weights(self, path):
        path += '_LR_{}'.format(self.lr)
        self.actor.save(path)
        self.critic.save(path)

    def load_weights(self, path_actor, path_critic):
        self.critic.load_weights(path_critic)
        self.actor.load_weights(path_actor)