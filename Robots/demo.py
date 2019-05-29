import torch
from actor_critic import Actor
import gym
import numpy as np
import argparse

DEMO_LENGHT = 10


parser = argparse.ArgumentParser()
parser.add_argument("--env", help="Environment")
parser.add_argument("--her_strat", help="her strategy")
args = parser.parse_args()
if not args.her_strat or not args.env :
    print("Please give all the parameters")
    print("Example run: python3 demo.py --env FetchPush-v1 --her_strat final")
    exit(1)

if args.her_strat not in ['future', 'final']:
    print("Invalid HER strategy")
    print("Strategies: final, future, episode")
    exit(1)

# process the inputs
def process_inputs(o, g, o_mean, o_std, g_mean, g_std):
    o_clip = np.clip(o, -200, 200)
    g_clip = np.clip(g, -200, 200)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -5, 5)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -5, 5)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs

if __name__ == '__main__':
    # load the model param
    model_path = 'saved_models/%s/%s/model.pt' % (args.env, args.her_strat)
    o_mean, o_std, g_mean, g_std, model = torch.load(model_path, map_location=lambda storage, loc: storage)
    # create the environment
    env = gym.make("FetchReach-v1")
    # get the env param
    obs = env.reset()
    # get the environment params
    act_dim = env.action_space.shape[0]
    env_dim = obs['observation'].shape[0] + obs['desired_goal'].shape[0]
    act_range = env.action_space.high[0]
    # create the actor network
    actor_network = Actor(env_dim, act_dim, act_range)
    actor_network.load_state_dict(model)
    actor_network.eval()
    for i in range(DEMO_LENGHT):
        observation = env.reset()
        # start to do the demo
        obs = observation['observation']
        g = observation['desired_goal']
        for t in range(env._max_episode_steps):
            env.render()
            inputs = process_inputs(obs, g, o_mean, o_std, g_mean, g_std)
            with torch.no_grad():
                pi = actor_network(inputs)
            action = pi.detach().numpy().squeeze()
            # put actions into the environment
            observation_new, reward, done, info = env.step(action)
            if info['is_success']:
                break
            obs = observation_new['observation']
            
        print('the episode is: {}, is success: {}'.format(i, info['is_success']))
