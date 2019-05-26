from DDPG import DDPG
import gym

env = gym.make("FetchPush-v1")

print(env.action_space)

