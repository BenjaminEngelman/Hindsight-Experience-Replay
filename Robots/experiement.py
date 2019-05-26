from DDPG import DDPG
import gym
import tensorflow as tf
env = gym.make("FetchPush-v1")
env2 = gym.make("LunarLander-v2")

obs = env.reset()


act_dim = env.action_space.shape[0]
env_dim = obs['observation'].shape[0] + obs['desired_goal'].shape[0]
act_range = env.action_space.high

print(act_dim, env_dim, act_range)

agent  = DDPG(act_dim, env_dim, act_range, 1, buffer_size=1000000, gamma=0.98)

# TRAIN THE FUCKING SHITTTTTTT
summary_writer = tf.summary.FileWriter("logs_DDPG")
train_args = {
    "render": True,
    "nb_episodes": 250 * 16,
    "batch_size": 128,
    "gather_stats": False,

}

agent.train(env, train_args, summary_writer)