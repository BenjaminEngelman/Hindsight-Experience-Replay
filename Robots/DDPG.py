import sys
import numpy as np
import time
from tqdm import tqdm
from actor_critic import Actor, Critic
from memory_buffer import MemoryBuffer
from normalizer import Normalizer
import torch
import os


NUM_EPOCHS = 200
NUM_CYCLES = 10
NUM_EPISODES = 16
OPTIMIZATION_STEPS = 40
NUM_TEST = 10


class DDPG:
    """ Deep Deterministic Policy Gradient (DDPG) Helper Class
    """

    def __init__(self, env, act_dim, state_dim, goal_dim, act_range, buffer_size=int(1e6), gamma=0.98, lr=0.001, tau=0.95):
        """ Initialization
        """
        # Environment and A2C parameters
        self.act_dim = act_dim
        self.act_range = act_range
        self.env_dim = state_dim + goal_dim
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.env = env

        # Create actor and critic networks
        self.actor_network = Actor(self.env_dim, act_dim, act_range)
        self.actor_target_network = Actor(self.env_dim, act_dim, act_range)
        self.critic_network = Critic(self.env_dim, act_dim, act_range)
        self.critic_target_network = Critic(self.env_dim, act_dim, act_range)

        # Optimizer
        self.actor_optim = torch.optim.Adam(
            self.actor_network.parameters(), lr=lr)
        self.critic_optim = torch.optim.Adam(
            self.critic_network.parameters(), lr=lr)

        # Replay buffer
        self.buffer = MemoryBuffer(buffer_size)

        # Normalizers
        self.goal_normalizer = Normalizer(
            goal_dim, default_clip_range=5)  # Clip between [-5, 5]
        self.state_normalizer = Normalizer(state_dim, default_clip_range=5)

    def policy_action(self, s, g):
        """ Use the actor to predict value
        """
        input = self.preprocess_inputs(s, g)
        return self.actor_network(input)

    def memorize(self, experiences):
        """ Store experience in memory buffer
        """
        for exp in experiences:
            state, action, reward, done, new_state, ag, goal = exp
            self.buffer.memorize(np.copy(state), np.copy(
                action), reward, done, np.copy(new_state), np.copy(ag), np.copy(goal))

    def sample_batch(self, batch_size):
        return self.buffer.sample_batch(batch_size)

    def clip_states_goals(self, state, goal):
        state = np.clip(state, -200, 200)
        goal = np.clip(goal, -200, 200)
        return state, goal

    def preprocess_inputs(self, state, goal):
        """Normalize and concatenate state and goal"""
        #state, goal = self.clip_states_goals(state, goal)
        state_norm = self.state_normalizer.normalize(state)
        goal_norm = self.goal_normalizer.normalize(goal)
        inputs = np.concatenate([state_norm, goal_norm])
        return torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)

    def select_actions(self, pi):
        # add the gaussian
        action = pi.cpu().numpy().squeeze()
        action += 0.2 * self.act_range * np.random.randn(*action.shape)
        action = np.clip(action, -self.act_range, self.act_range)
        # random actions...
        random_actions = np.random.uniform(low=-self.act_range, high=self.act_range,
                                           size=self.act_dim)
        # choose if use the random actions
        action += np.random.binomial(1, 0.3, 1)[0] * (random_actions - action)
        return action

    def update_network(self, batch_size):
        s, actions, rewards, _, ns, _, g, _ = self.sample_batch(batch_size)
        # DAV UPDATE
        # Preprocess
        states, goals = self.clip_states_goals(s, g)
        new_states, new_goals = self.clip_states_goals(ns, g)

        norm_states = self.state_normalizer.normalize(states)
        norm_goals = self.goal_normalizer.normalize(goals)
        inputs_norm = np.concatenate([norm_states, norm_goals], axis=1)

        norm_new_states = self.state_normalizer.normalize(new_states)
        norm_new_goals = self.goal_normalizer.normalize(new_goals)
        inputs_next_norm = np.concatenate(
            [norm_new_states, norm_new_goals], axis=1)

        # To tensor
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(
            inputs_next_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.float32)
        r_tensor = torch.tensor(rewards, dtype=torch.float32)

        with torch.no_grad():
            # do the normalization
            # concatenate the stuffs
            actions_next = self.actor_target_network(inputs_next_norm_tensor)
            q_next_value = self.critic_target_network(
                inputs_next_norm_tensor, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + self.gamma * q_next_value
            target_q_value = target_q_value.detach()
            # clip the q value
            clip_return = 1 / (1 - self.gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)
        # the q loss
        real_q_value = self.critic_network(inputs_norm_tensor, actions_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()
        # the actor loss
        actions_real = self.actor_network(inputs_norm_tensor)
        actor_loss = - \
            self.critic_network(inputs_norm_tensor, actions_real).mean()
        actor_loss += 1 * (actions_real / self.act_range).pow(2).mean()
        # start to update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        # update the critic_network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

    def soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                (1 - self.tau) * param.data + self.tau * target_param.data)

    def train(self, args):
        self.create_save_dir(
            args["save_dir"], args["env_name"], args["HER_strat"])

        success_rates = []
        for ep_num in range(NUM_EPOCHS):
            start = time.time()
            for cycle in range(NUM_CYCLES):
                print("Cycle : %d" % cycle)
                for _ in range(NUM_EPISODES):
                    # Reset episode
                    observation = self.env.reset()
                    old_state = observation['observation']
                    old_achieved_goal = observation['achieved_goal']
                    goal = observation['desired_goal']
                    episode_exp = []
                    episode_exp_her = []
                    done = False
                    for _ in range(args["max_timesteps"]):
                        if args['render']:
                            self.env.render()
                        with torch.no_grad():
                            # Actor picks an action (following the deterministic policy)
                            pi = self.policy_action(old_state, goal)
                            action = self.select_actions(pi)
                        # Retrieve new state, reward, and whether the state is terminal
                        obs, reward, done, _ = self.env.step(action)
                        new_state = obs['observation']
                        new_achieved_goal = obs['achieved_goal']
                        # Add outputs to memory buffer
                        episode_exp.append([old_state.copy(), action.copy(), reward, done, new_state.copy(), old_achieved_goal.copy(), goal.copy()])

                        old_achieved_goal = new_achieved_goal
                        old_state = new_state

                    if args["HER_strat"] == "final":
                        experience = episode_exp[-1]
                        experience[3] = True  # set done = true
                        # set g' to achieved goal
                        experience[-1] = np.copy(experience[-2])
                        reward = self.env.compute_reward(
                            experience[-2], experience[-1], None)  # set reward of success
                        experience[2] = reward
                        episode_exp_her.append(experience)

                    elif args["HER_strat"] in ["future", "episode"]:
                        # For each transition of the episode trajectory
                        for t in range(len(episode_exp)):
                            # Add K random states which come from the same episode as the transition
                            for _ in range(args["HER_k"]):
                                if args["HER_strat"] == "future":
                                    # Select a future exp from the same episod
                                    selected = np.random.randint(
                                        t, len(episode_exp))
                                elif args["HER_strat"] == "episode":
                                    # Select an exp from the same episode
                                    selected = np.random.randint(
                                        0, len(episode_exp))
                                # Take the achieved goal of the selected
                                ag_selected = np.copy(episode_exp[selected][5])
                                s, a, _, d, ns, ag, _ = episode_exp[t]
                                r = self.env.compute_reward(
                                    ag_selected, ag, None)
                                # New transition where the achieved goal of the selected is the new goal
                                her_transition = [np.copy(s), np.copy(a), r, d, np.copy(
                                    ns), np.copy(ag), np.copy(ag_selected)]
                                episode_exp_her.append(her_transition)

                    self.memorize(episode_exp)
                    self.memorize(episode_exp_her)

                    # Update Normalizers with the observations of this episode
                    self.update_normalizers(episode_exp, episode_exp_her)

                # Train network
                for _ in range(OPTIMIZATION_STEPS):
                    # Sample experience from buffer
                    self.update_network(args["batch_size"])

                # Soft update the target networks
                self.soft_update_target_network(
                    self.actor_target_network, self.actor_network)
                self.soft_update_target_network(
                    self.critic_target_network, self.critic_network)

            success_rate = self.eval()
            success_rates.append(success_rate)
            print("Epoch:", ep_num+1, " -- success rate:",
                  success_rates[-1], " -- duration:", time.time() - start)
            torch.save([self.state_normalizer.mean, self.state_normalizer.std, self.goal_normalizer.mean, self.goal_normalizer.std, self.actor_network.state_dict()],
                       self.model_path + '/model.pt')

        return success_rates

    def create_save_dir(self, save_dir, env_name, her_strat):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        # path to save the model
        subdir = os.path.join(save_dir, env_name)
        if not os.path.exists(subdir):
            os.mkdir(subdir)
        self.model_path = os.path.join(save_dir, env_name, her_strat)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

    def update_normalizers(self, episode_exp, episode_exp_her):
        # Update Normalizers
        episode_exp_states = np.vstack(np.array(episode_exp)[:, 0])
        episode_exp_goals = np.vstack(np.array(episode_exp)[:, 6])
        if len(episode_exp_her) != 0 :
            episode_exp_her_states = np.vstack(np.array(episode_exp_her)[:, 0])
            episode_exp_her_goals = np.vstack(np.array(episode_exp_her)[:, 6])
            states = np.concatenate([episode_exp_states, episode_exp_her_states])
            goals = np.concatenate([episode_exp_goals, episode_exp_her_goals])
        else:
            states = np.copy(episode_exp_states)
            goals = np.copy(episode_exp_goals)

        states, goals = self.clip_states_goals(states, goals)
        self.state_normalizer.update(states)
        self.goal_normalizer.update(goals)
        self.state_normalizer.recompute_stats()
        self.goal_normalizer.recompute_stats()

    def eval(self):
        print("Evaluation")
        eval_success = 0
        for _ in range(NUM_TEST):
            observation = self.env.reset()
            state = observation['observation']
            goal = observation['desired_goal']
            for _ in range(self.env._max_episode_steps):
                with torch.no_grad():
                    input = self.preprocess_inputs(state, goal)
                    pi = self.actor_network(input)
                    action = pi.detach().cpu().numpy().squeeze()
                new_observation, _, _, info = self.env.step(action)
                state = new_observation['observation']
            if info['is_success']:
                eval_success += 1

        eval_success = eval_success / NUM_TEST
        return eval_success

    # def save_weights(self, path):
    #     path += '_LR_{}'.format(self.lr)
    #     self.actor.save(path + "_actor")
    #     self.critic.save(path + "_critic")

    # def load_weights(self, path_actor, path_critic):
    #     self.critic.load_weights(path_critic)
    #     self.actor.load_weights(path_actor)
