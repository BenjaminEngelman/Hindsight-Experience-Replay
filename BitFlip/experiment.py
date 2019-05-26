from Environment import BitFlippingEnv
from DQN import DQN
from ReplayMemory import ReplayMemory
import matplotlib.pyplot as plt
import time
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--her", help="her strategy")
args = parser.parse_args()
if not args.her:
    print("Please give the HER strategy")
    exit(1)

HER_STRATEGY = args.her
K = 4 # For strategy future
REPLAY_MEMORY_SIZE = 100000
NUM_BITS = 15
NUM_EPOCHS = 250
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
            # Final experience becomes (s, a, r, s', g') where g' = s'
            experience = episode_exp[-1]
            experience[-2] = True # set done = true
            experience[-1] = experience[-3] # set g' = s'
            experience[2] = 1 # set reward of success
            episode_exp_her.append(experience)

        elif HER_STRATEGY in ["future", "episode"]:
            # For each transition of the episode trajectory
            for t in range(len(episode_exp)):
                # Add K random states which come from the same episode as the transition
                for k in range(K):
                    if HER_STRATEGY == "future":
                        # Select a future exp from the same episod
                        selected = np.random.randint(t, len(episode_exp)) 
                    elif HER_STRATEGY == "episode":
                        # Select an exp from the same episode
                        selected = np.random.randint(0, len(episode_exp))  
                    _, _, _, g, _, _ = episode_exp[selected] # g = s' of selected
                    s, a, _, ns, _, _  = episode_exp[t]
                    d = np.array_equal(ns, g)
                    r = 1 if d else -1
                    episode_exp_her.append([s, a, r, ns, d, g])


        agent.remember(episode_exp)
        agent.remember(episode_exp_her)


    agent.replay(OPTIMIZATION_STEPS)
    agent.next_episode(n)
    
    success_rate.append(successes/NUM_EPISODES)
    print("Epoch:", i+1, " -- success rate:", success_rate[-1], " -- epsilon: ", agent.epsilon)

print("Training time : %.2f"%(time.clock()-start), "s")

with open('results/BitFlip_HER_%s.txt'%(HER_STRATEGY), 'w') as f:
    for item in success_rate:
        f.write("%s\n" % item)

plt.plot(success_rate)
plt.title("Success rate by epoch using HER with %s strategy" % (HER_STRATEGY))
plt.xlabel("epoch")
plt.ylabel("Success rate")
plt.savefig("plots/BitFlip_HER_%s.svg" % (HER_STRATEGY), format="svg")
