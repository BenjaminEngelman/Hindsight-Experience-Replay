from DDPG import DDPG
import gym
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt

# FetchPush-v1

parser = argparse.ArgumentParser()
parser.add_argument("--env", help="Environment")
parser.add_argument("--her_strat", help="her strategy")
parser.add_argument("--k", help="number of additionnal goals used to replay each transition with")
parser.add_argument("--render", help="render or not")


args = parser.parse_args()
if not args.her_strat or not args.env or not args.k:
    print("Please give all the parameters")
    print("Example run: python3 experiment.py --env FetchPush-v1 --her_strat future --k 4 --render False")
    exit(1)

if args.her_strat not in ['future', 'final']:
    print("Invalid HER strategy")
    print("Strategies: final, future, episode")
    exit(1)

env_name = args.env
env = gym.make(env_name)
obs = env.reset()

act_dim = env.action_space.shape[0]
state_dim = obs['observation'].shape[0]
goal_dim = obs['desired_goal'].shape[0]
act_range = env.action_space.high[0]

agent  = DDPG(env, act_dim, state_dim, goal_dim, act_range, buffer_size=1000000, gamma=0.98)
train_args = {
    "env_name": args.env,
    "save_dir": "saved_models",
    "render": False if args.render == "False" else True,
    "batch_size": 256,
    "HER_strat": args.her_strat,
    "HER_k": int(args.k),
    "max_timesteps": env._max_episode_steps
}

success_rate = agent.train(train_args)


#### SAVE RESULTS #### 

with open('results/%s_%s_k=%s.txt'%(env_name, args.her_strat, args.k), 'w') as f:
    for item in success_rate:
        f.write("%s\n" % item)

plt.plot(success_rate)
plt.title("%s with %s strategy (k=%s)" % (env_name, args.her_strat, args.k))
plt.xlabel("epoch")
plt.ylabel("Success rate")
plt.savefig("plots/%s_%s_k=%s.svg" % (env_name, args.her_strat, args.k), format="svg")