import sys
import numpy as np

from tqdm import tqdm
from actor import Actor
from critic import Critic
from stats import gather_stats
from networks import tfSummary, OrnsteinUhlenbeckProcess
from memory_buffer import MemoryBuffer

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

    def memorize(self, state, action, reward, done, new_state, goal):
        """ Store experience in memory buffer
        """
        self.buffer.memorize(state, action, reward, done, new_state, goal)

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

    def train(self, env, args, summary_writer):
        results = []

        # First, gather experience
        tqdm_e = tqdm(range(args['nb_episodes']), desc='Score', leave=True, unit=" episodes")
        for e in tqdm_e:

            # Reset episode
            time, cumul_reward, done = 0, 0, False

            observation = env.reset()
            old_state = observation['observation']
            ag = observation['achieved_goal']
            goal = observation['desired_goal']

            actions, states, rewards = [], [], []
            noise = OrnsteinUhlenbeckProcess(size=self.act_dim)

            while not done:
                if args['render']: env.render()
                # Actor picks an action (following the deterministic policy)
                a = self.policy_action(old_state, goal)
                # Clip continuous values to be valid w.r.t. environment
                a = np.clip(a+noise.generate(time), -self.act_range, self.act_range)
                # Retrieve new state, reward, and whether the state is terminal
                obs, r, done, _ = env.step(a)
                new_state = obs['observation']
                ag = obs['achieved_goal']
                # Add outputs to memory buffer
                self.memorize(old_state, a, r, done, new_state, goal)
                # Sample experience from buffer
                states, actions, rewards, dones, new_states, goals,_ = self.sample_batch(args["batch_size"])
                inputs = np.concatenate((new_states, goals), axis=1)
                q_values = self.critic.target_predict([inputs, self.actor.target_predict(inputs)])
                # Compute critic target
                critic_target = self.bellman(rewards, q_values, dones)
                # Train both networks on sampled batch, update target networks
                inputs = np.concatenate((states, goals), axis=1)

                self.update_models(inputs, actions, critic_target)
                # Update current state
                old_state = new_state
                cumul_reward += r
                time += 1

            # Gather stats every episode for plotting
            if(args["gather_stats"]):
                mean, stdev = gather_stats(self, env)
                results.append([e, mean, stdev])

            # Export results for Tensorboard
            score = tfSummary('score', cumul_reward)
            summary_writer.add_summary(score, global_step=e)
            summary_writer.flush()
            # Display score
            tqdm_e.set_description("Score: " + str(cumul_reward))
            tqdm_e.refresh()

        return results

    def save_weights(self, path):
        path += '_LR_{}'.format(self.lr)
        self.actor.save(path)
        self.critic.save(path)

    def load_weights(self, path_actor, path_critic):
        self.critic.load_weights(path_critic)
        self.actor.load_weights(path_actor)