from Environment import BitFlippingEnv
from DQN import DQN
from ReplayMemory import ReplayMemory
import matplotlib.pyplot as plt
import time
import numpy as np

HER_STRATEGY = "future"
K = 4 # For strategy future
REPLAY_MEMORY_SIZE = 100000
NUM_BITS = 15
NUM_EPOCHS = 2000
NUM_EPISODES = 16
OPTIMIZATION_STEPS = 40

success_rate = []

env = BitFlippingEnv(n=NUM_BITS)
agent = DQN(num_action=NUM_BITS, state_size=NUM_BITS, goal_size=NUM_BITS)


start = time.clock()
for i in range(NUM_EPOCHS):
    successes = 0
    for n in range(NUM_EPISODES):
        episode_exp = [] 
        episode_exp_her = []

        state, goal = env.reset()
        for t in range(NUM_BITS):
            action = agent.get_action(state, goal)
            next_state, reward, done= env.step(action)
            episode_exp.append([state, action, reward, next_state, done, goal])
            state = next_state
            if done:
                break
        successes += done

        if HER_STRATEGY == "final":
            experience = episode_exp[-1]
            experience[-1] = state # substitute the goal
            experience[-2] = True # set done = true
            experience[2] = 1 # set reward of success
            episode_exp.append(experience)

        elif HER_STRATEGY == "future": # The strategy can be changed here
            # For each transition of the episode trajectory
            for t in range(len(episode_exp)):
                # Add K random states which come from the same episode as the transition
                for k in range(K):
                    # Select a number between t and the lenght of the episode 
                    future = np.random.randint(t, len(episode_exp))
                    g = episode_exp[future][3] # next_state of future
                    s = episode_exp[t][0] # state of future
                    a = episode_exp[t][1] # action of future
                    ns = episode_exp[t][3] # next state of future
                    d = np.array_equal(ns, g)
                    r = 1 if done else -1
                    episode_exp_her.append([s, a, r, ns, d, g])
        

        agent.remember(episode_exp)
        agent.remember(episode_exp_her)


    agent.replay(OPTIMIZATION_STEPS)
    agent.next_episode(n)
    
    success_rate.append(successes/NUM_EPISODES)
    print("Epoch:", i+1, " -- success rate:", success_rate[-1])

print("Training time : %.2f"%(time.clock()-start), "s")

with open('results/BitFlip_HER_FINAL.txt', 'w') as f:
    for item in success_rate:
        f.write("%s\n" % item)

plt.plot(range(0, NUM_EPOCHS), success_rate)
plt.title("Success rate by epoch using HER with final strategy")
plt.xlabel("epoch")
plt.ylabel("Success rate")
plt.savefig("plots/BitFlip_HER_FINAL.svg", format="svg")
