from Environment import BitFlippingEnv
from DQN import DQN
from ReplayMemory import ReplayMemory
import matplotlib.pyplot as plt
import time

HER_STRATEGY = "final"
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
            next_state, reward, done = env.step(action)
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

        # if HER_STRATEGY == "future": # The strategy can be changed here
        #     #         goal = state # HER, with substituted goal=final_state
        #     for t in range(len(episode_memory.memory)):
        #         for k in range(K):
        #             future = np.random.randint(t, len(episode_memory.memory))
        #             goal = episode_memory.memory[future][3] # next_state of future
        #             state = episode_memory.memory[t][0]
        #             action = episode_memory.memory[t][1]
        #             next_state = episode_memory.memory[t][3]
        #             done = np.array_equal(next_state, goal)
        #             reward = 0 if done else -1
        #             episode_memory_her.add(state, action, reward, next_state, done, goal)
        

        agent.remember(episode_exp)
        agent.remember(episode_exp_her)


    agent.replay(OPTIMIZATION_STEPS)
    agent.next_episode(n)
    
    success_rate.append(successes/NUM_EPISODES)
    print("epoch", i+1, "success rate", success_rate[-1])

print("Training time : %.2f"%(time.clock()-start), "s")

with open('results/BitFlip_HER_FINAL.txt', 'w') as f:
    for item in success_rate:
        f.write("%s\n" % item)

plt.plot(range(0, NUM_EPOCHS), success_rate)
plt.title("Success rate by epoch using HER with final strategy")
plt.xlabel("epoch")
plt.ylabel("Success rate")
plt.savefig("plots/BitFlip_HER_FINAL.svg", format="svg")
