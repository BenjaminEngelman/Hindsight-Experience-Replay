from helper import parallelize, saveComplexJson
from Environment import BitFlippingEnv
from experiment import learn
from DQN import DQN

NUM_PROCESSES = 32
NUM_TRIAL = 5

NUM_BITS = 15


STRATS = ['none', 'final', 'future', 'episode']

def runTrial(her_strat, per):
    env = BitFlippingEnv(n=NUM_BITS)
    agent = DQN(num_action=NUM_BITS, state_size=NUM_BITS, goal_size=NUM_BITS, PER=per)
    success = learn(agent, env, her_strat)

    return success


def addJob(jobs, pool):
    for i in range(NUM_TRIAL):
        for strat in STRATS :
            for per in [
                False,
                # True
            ]:
                name = "%s_%s" % (strat, per)
                jobs[(name, i)] = pool.apply_async(runTrial, (strat, per))

if __name__ == "__main__":
    resultsFilename = "results/clusterRes.json"
    results = parallelize(addJob, numProcesses=NUM_PROCESSES)
    saveComplexJson(resultsFilename, results)
    print("Done.")