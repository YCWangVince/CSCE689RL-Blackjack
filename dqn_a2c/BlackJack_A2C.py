# build_parser, readCommand, parse_list functions are adopted from the code of homework of CSCE689's run.py file.
import numpy as np
import gym
import rlcard
from Solvers.Abstract_Solver_proj import AbstractSolver, Statistics
from Solvers.A2C_proj import A2C
#from Solvers.DQN_proj import DQN
from Solvers.DQN import DQN
import optparse
import sys
#import Solvers.Available_solvers as avs
import random
from Solvers import plotting

def build_parser():
    parser = optparse.OptionParser(description='Run a specified RL algorithm on a specified domain.')
    #parser.add_option("-s", "--solver", dest="solver", type="string", default="random",
    #                  help='Solver from ' + str(avs.solvers))
    parser.add_option("-d", "--domain", dest="domain", type="string", default="Gridworld",
                      help='Domain from OpenAI Gym')
    parser.add_option("-o", "--outfile", dest="outfile", default="out",
                      help="Write results to FILE", metavar="FILE")
    parser.add_option("-x", "--experiment_dir", dest="experiment_dir", default="Experiment",
                      help="Directory to save Tensorflow summaries in", metavar="FILE")
    parser.add_option("-e", "--episodes", type="int", dest="episodes", default=500,
                      help='Number of episodes for training')
    parser.add_option("-t", "--steps", type="int", dest="steps", default=10000,
                      help='Maximal number of steps per episode')
    parser.add_option("-l", "--layers", dest="layers", type="string", default="[24,24]",
                      help='size of hidden layers in a Deep neural net. e.g., "[10,15]" creates a net where the'
                           'Input layer is connected to a layer of size 10 that is connected to a layer of size 15'
                           ' that is connected to the output')
    parser.add_option("-a", "--alpha", dest="alpha", type="float", default=0.001,
                      help='The learning rate (alpha) for updating state/action values')
    parser.add_option("-r", "--seed", type="int", dest="seed", default=random.randint(0, 9999999999),
                      help='Seed integer for random stream')
    parser.add_option("-G", "--graphics", type="int", dest="graphics_every", default=0,
                      help='Graphic rendering every i episodes. i=0 will present only one, post training episode.'
                           'i=-1 will turn off graphics. i=1 will present all episodes.')
    parser.add_option("-g", "--gamma", dest="gamma", type="float", default=1.00,
                      help='The discount factor (gamma)')
    parser.add_option("-p", "--epsilon", dest="epsilon", type="float", default=0.1,
                      help='Initial epsilon for epsilon greedy policies (might decay over time)')
    parser.add_option("-P", "--final_epsilon", dest="epsilon_end", type="float", default=0.1,
                      help='The final minimum value of epsilon after decaying is done')
    parser.add_option("-c", "--decay", dest="epsilon_decay", type="float", default=0.99,
                                        help='Epsilon decay factor')
    parser.add_option("-m", "--replay", type="int", dest="replay_memory_size", default=500000,
                      help='Size of the replay memory')
    parser.add_option("-N", "--update", type="int", dest="update_target_estimator_every", default=10000,
                      help='Copy parameters from the Q estimator to the target estimator every N steps.')
    parser.add_option("-b", "--batch_size", type="int", dest="batch_size", default=32,
                      help='Size of batches to sample from the replay memory')
    parser.add_option('--no-plots', help='Option to disable plots if the solver results any',
            dest = 'disable_plots', default = False, action = 'store_true')
    return parser

def readCommand(argv):
    parser = build_parser()
    (options, args) = parser.parse_args(argv)
    return options

def parse_list(string):
    string.strip()
    string = string[1:-1].split(',') # Change "[0,1,2,3]" to '0', '1', '2', '3'
    l = []
    for n in string:
        l.append(int(n))
    return l


class BlackjackEnv(gym.Env):
    def __init__(self):
        self._rlcard_env = rlcard.make('blackjack')
        self.action_space = gym.spaces.Discrete(self._rlcard_env.num_actions)
        self.observation_space = gym.spaces.Box(0, 31, shape=self._rlcard_env.state_shape[0], dtype=np.int32)

    def seed(self, seed):
        self._rlcard_env.seed(seed)

    def reset(self):
        obs, _ = self._rlcard_env.reset()
        obs = tuple(obs["obs"].tolist())
        return obs

    def step(self, action):
        obs, _ = self._rlcard_env.step(action)
        obs = tuple(obs["obs"].tolist())
        done = False
        reward = 0.0
        if self._rlcard_env.is_over():
            done = True
            reward = float(self._rlcard_env.get_payoffs()[0])
        #print(reward)
        return obs, reward, done, {}

if __name__ == "__main__":
    options = readCommand(sys.argv)
    options.layers = parse_list(options.layers)
    env = BlackjackEnv()
    solver = A2C(env, options)
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(options.episodes),
        episode_rewards=np.zeros(options.episodes))
    average_reward = 0
    average_window = 1000
    list_average_reward = []
    list_window_reward = []
    window_reward = []
    for i_episode in range(options.episodes):
        solver.init_stats()
        solver.statistics[Statistics.Episode.value] += 1
        env.reset()
        solver.train_episode()
        solver.render = False
        #result_file.write(solver.get_stat() + '\n')
        if options.epsilon > options.epsilon_end:
            options.epsilon *= options.epsilon_decay
        stats.episode_rewards[i_episode] = solver.statistics[Statistics.Rewards.value]
        stats.episode_lengths[i_episode] = solver.statistics[Statistics.Steps.value]
        average_reward += (1. / (i_episode + 1)) * (solver.statistics[Statistics.Rewards.value] - average_reward)
        list_average_reward.append(average_reward)
        if len(window_reward) < 99:
            window_reward.append(solver.statistics[Statistics.Rewards.value])
        else:
            #average_reward.append(stats.episode_rewards[i_episode])
            #print("Average reward {}".format(np.mean(average_reward)))
            print("Episode {}: Average reward {}".format(i_episode+1, np.mean(window_reward)))
            list_window_reward.append(np.mean(window_reward))
            window_reward = []
            #average_reward = []
        #print("Episode {}: Reward {}, Steps {}".format(i_episode+1, solver.statistics[Statistics.Rewards.value], solver.statistics[Statistics.Steps.value]))
    np.save('A2C.npy', np.array(list_window_reward))
    solver.close()

