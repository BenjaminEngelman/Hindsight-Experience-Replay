import sys
import numpy as np
import time
from actor_critic import Actor, Critic
from ReplayMemory import ReplayMemory
from normalizer import Normalizer
import torch
import os
from utils import *
from mpi4py import MPI
from copy import deepcopy


NUM_EPOCHS = 200
NUM_CYCLES = 10
NUM_EPISODES = 16
ROLLOUT_PER_WORKER = 2
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
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())

        self.critic_network = Critic(self.env_dim, act_dim, act_range)
        self.critic_target_network = Critic(self.env_dim, act_dim, act_range)
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())

        sync_networks(self.actor_network)
        sync_networks(self.critic_network)


        # Optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=lr)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=lr)

        # Replay buffer
        # self.buffer = MemoryBuffer(buffer_size)
        self.buffer = ReplayMemory(buffer_size)

        # Normalizers
        self.goal_normalizer = Normalizer(goal_dim, default_clip_range=5)  # Clip between [-5, 5]
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
            self.buffer.push(exp)

    def sample_batch(self, batch_size):
        return deepcopy(self.buffer.sample(batch_size))

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
        action = np.clip(action, -self.act_range, self.act_range)

        return action

    def update_network(self, batch_size):
        s, actions, rewards, ns, _, g = self.sample_batch(batch_size)
        
        states, goals = self.clip_states_goals(s, g)
        new_states, new_goals = self.clip_states_goals(ns, g)

        norm_states = self.state_normalizer.normalize(states)
        norm_goals = self.goal_normalizer.normalize(goals)
        inputs_norm = np.concatenate([norm_states, norm_goals], axis=1)

        norm_new_states = self.state_normalizer.normalize(new_states)
        norm_new_goals = self.goal_normalizer.normalize(new_goals)
        inputs_next_norm = np.concatenate([norm_new_states, norm_new_goals], axis=1)

        # To tensor
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.float32)
        r_tensor = torch.tensor(rewards, dtype=torch.float32)

        with torch.no_grad():
            # do the normalization
            # concatenate the stuffs
            actions_next = self.actor_target_network(inputs_next_norm_tensor)
            q_next_value = self.critic_target_network(inputs_next_norm_tensor, actions_next)
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
        actor_loss = - self.critic_network(inputs_norm_tensor, actions_real).mean()
        actor_loss += 1.0 * (actions_real / self.act_range).pow(2).mean()
        # start to update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor_network)
        self.actor_optim.step()
        # update the critic_network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        sync_grads(self.critic_network)
        self.critic_optim.step()

    def soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.tau) * param.data + self.tau * target_param.data)

    def train(self, args):
        if MPI.COMM_WORLD.Get_rank() == 0:
            self.create_save_dir(args["save_dir"], args["env_name"], args["HER_strat"])

        success_rates = []
        for ep_num in range(NUM_EPOCHS):
            start = time.time()
            for _ in range(NUM_CYCLES):
                for _ in range(ROLLOUT_PER_WORKER):
                    # Reset episode
                    observation = self.env.reset()
                    current_state = observation['observation']
                    goal = observation['desired_goal']
                    old_achieved_goal = observation['achieved_goal']
                    episode_exp = []
                    episode_exp_her = []
                    for _ in range(self.env._max_episode_steps):
                        if args['render']: self.env.render()
                        with torch.no_grad():
                            pi = self.policy_action(current_state, goal)
                            action = self.select_actions(pi)
                        obs, reward, _, _ = self.env.step(action)
                        new_state = obs['observation']
                        new_achieved_goal = obs['achieved_goal']
                        # Add outputs to memory buffer
                        episode_exp.append([current_state, action, reward, new_state, old_achieved_goal, goal])
                        if reward == 0 : break

                        old_achieved_goal = new_achieved_goal
                        current_state = new_state

                    if args["HER_strat"] == "final":
                        experience = episode_exp[-1]
                        # set g' to achieved goal
                        experience[-1] = np.copy(experience[-2])
                        reward = self.env.compute_reward(experience[-2], experience[-1], None)  # set reward of success
                        experience[2] = reward
                        episode_exp_her.append(experience)

                    elif args["HER_strat"] in ["future", "episode"]:
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
                                # Take the achieved goal of the selected
                                ag_selected = np.copy(episode_exp[selected][5])
                                s, a, _, ns, ag, _ = episode_exp[t]
                                r = self.env.compute_reward(ag_selected, ag, None)
                                # New transition where the achieved goal of the selected is the new goal
                                her_transition = [s, a, r, ns, ag, ag_selected]
                                episode_exp_her.append(her_transition)

                    self.memorize(deepcopy(episode_exp))
                    self.memorize(deepcopy(episode_exp_her))

                    # Update Normalizers with the observations of this episode
                    self.update_normalizers(deepcopy(episode_exp), deepcopy(episode_exp_her))

                for _ in range(OPTIMIZATION_STEPS):
                    # Sample experience from buffer
                    self.update_network(args["batch_size"])

                # Soft update the target networks
                self.soft_update_target_network( self.actor_target_network, self.actor_network)
                self.soft_update_target_network( self.critic_target_network, self.critic_network)

            success_rate = self.eval()
            success_rates.append(success_rate)
            if MPI.COMM_WORLD.Get_rank() == 0:
                print("Epoch:", ep_num+1, " -- success rate:", success_rates[-1], " -- duration:", time.time() - start)
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
        episode_exp_goals = np.vstack(np.array(episode_exp)[:, 5])
        if len(episode_exp_her) != 0 :
            episode_exp_her_states = np.vstack(np.array(episode_exp_her)[:, 0])
            episode_exp_her_goals = np.vstack(np.array(episode_exp_her)[:, 5])
            states = np.concatenate([episode_exp_states, episode_exp_her_states])
            goals = np.concatenate([episode_exp_goals, episode_exp_her_goals])
        else:
            states = np.copy(episode_exp_states)
            goals = np.copy(episode_exp_goals)

        states, goals = self.clip_states_goals(states, goals)

        self.state_normalizer.update(deepcopy(states))
        self.goal_normalizer.update(deepcopy(goals))
        self.state_normalizer.recompute_stats()
        self.goal_normalizer.recompute_stats()

    def eval(self):
        total_success_rate = []
        for _ in range(NUM_TEST):
            per_success_rate = []
            observation = self.env.reset()
            state = observation['observation']
            goal = observation['desired_goal']
            for _ in range(self.env._max_episode_steps):
                # self.env.render()
                with torch.no_grad():
                    input = self.preprocess_inputs(state, goal)
                    pi = self.actor_network(input)
                    action = pi.detach().cpu().numpy().squeeze()
                new_observation, _, _, info = self.env.step(action)
                state = new_observation['observation']
                per_success_rate.append(info['is_success'])
            total_success_rate.append(per_success_rate)
               
        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        return global_success_rate / MPI.COMM_WORLD.Get_size()