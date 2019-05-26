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

args = parser.parse_args()
if not args.her_strat or not args.env or not args.k:
    print("Please give all the parameters")
    print("Example run: python3 experiment.py --env FetchPush-v1 --her_strat future --k 4")
    exit(1)

env_name = args.env
env = gym.make(env_name)
obs = env.reset()

act_dim = env.action_space.shape[0]
env_dim = obs['observation'].shape[0] + obs['desired_goal'].shape[0]
act_range = env.action_space.high

agent  = DDPG(act_dim, env_dim, act_range, 1, buffer_size=1000000, gamma=0.98)
summary_writer = tf.summary.FileWriter("logs_DDPG")
train_args = {
    "render": False,
    "nb_episodes": 250 * 16,
    "batch_size": 128,
    "HER_strat": args.her_strat,
    "HER_k": int(args.k)
}

success_rate = agent.train(env, train_args)


#### SAVE RESULTS #### 

with open('results/%s_%s_k=%s.txt'%(env_name, args.her_strat, args.k), 'w') as f:
    for item in success_rate:
        f.write("%s\n" % item)

plt.plot(success_rate)
plt.title("%s with %s strategy (k=%s)" % (env_name, args.her_strat, args.k))
plt.xlabel("epoch")
plt.ylabel("Success rate")
plt.savefig("plots/%s_%s_k=%s.svg" % (env_name, args.her_strat, args.k), format="svg")