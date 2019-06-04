from Environment import BitFlippingEnv
from DQN import DQN
from ReplayMemory import ReplayMemory
import matplotlib.pyplot as plt
import time
import numpy as np
import argparse
from copy import deepcopy


# parser = argparse.ArgumentParser()
# parser.add_argument("--her_strat", help="her strategy")
# parser.add_argument("--per", help="Use PER")

# args = parser.parse_args()
# if not args.her_strat:
#     print("Please give the HER strategy")
#     exit(1)

# if not args.per:
#     print("Need PER arg (True or False)")
#     exit(1)

# HER_STRATEGY = args.her_strat
K = 4 # For strategy future
REPLAY_MEMORY_SIZE = 100000
NUM_BITS = 15
NUM_EPOCHS = 250
NUM_EPISODES = 16
OPTIMIZATION_STEPS = 40

# success_rate = []

# env = BitFlippingEnv(n=NUM_BITS)
# agent = DQN(num_action=NUM_BITS, state_size=NUM_BITS, goal_size=NUM_BITS, PER=args.per)

def learn(agent, env, her_strat):
    success_rate = []
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

            if her_strat == "final":
                # Final experience becomes (s, a, r, s', g') where g' = s'
                new_goal = deepcopy(episode_exp[-1][3]) # next state of last

                for experience in episode_exp:
                    her_exp = deepcopy(experience)
                    her_exp[-1] = new_goal
                    her_exp[-2] = np.array_equal(her_exp[3], new_goal)
                    her_exp[2] = 1 if her_exp[-2] else -1
                    episode_exp_her.append(her_exp)

            elif her_strat == "future":
                for t, experience in enumerate(episode_exp[0:-2]):
                    for _ in range(K):
                        selected = np.random.randint(t+1, len(episode_exp)) 

                        new_goal = deepcopy(episode_exp[selected][3]) # g = s' of selected
                        her_exp = deepcopy(episode_exp[selected])
                        her_exp[-1] = new_goal
                        her_exp[-2] = np.array_equal(her_exp[3], new_goal)
                        her_exp[2] = 1 if her_exp[-2] else -1
                        episode_exp_her.append(her_exp)

            elif her_strat == "episode":
                for t, experience in enumerate(episode_exp):
                    for _ in range(K):
                        selected = np.random.randint(0, len(episode_exp)) 

                        new_goal = deepcopy(episode_exp[selected][3]) # g = s' of selected
                        her_exp = deepcopy(episode_exp[selected])
                        her_exp[-1] = new_goal
                        her_exp[-2] = np.array_equal(her_exp[3], new_goal)
                        her_exp[2] = 1 if her_exp[-2] else -1
                        episode_exp_her.append(her_exp)


            agent.remember(deepcopy(episode_exp))
            agent.remember(deepcopy(episode_exp_her))


        agent.replay(OPTIMIZATION_STEPS)
        agent.next_episode(n)
        
        success_rate.append(successes/NUM_EPISODES)
        # print("Epoch:", i+1, " -- success rate:", success_rate[-1], " -- epsilon: ", agent.epsilon)
    
    return success_rate

    # print("Training time : %.2f"%(time.clock()-start), "s")

    # with open('results/BitFlip_HER_%s_PER_%s.txt'%(HER_STRATEGY, args.per), 'w') as f:
    #     for item in success_rate:
    #         f.write("%s\n" % item)

    # plt.plot(success_rate)
    # plt.title("Success rate by epoch using HER with %s strategy" % (HER_STRATEGY))
    # plt.xlabel("epoch")
    # plt.ylabel("Success rate")
    # plt.savefig("plots/BitFlip_HER_%s_PER_%s.svg" % (HER_STRATEGY, args.per), format="svg")
