import argparse
import numpy as np, os, time, sys, random
from core import repaly_memory
from core import ddpg
Environment_Name = 'Swimmer-v2'
import numpy as np
import torch,gym
from core import mod_neuro_evo as utils_ne
from core import mod_utils as utils


parser = argparse.ArgumentParser()
parser.add_argument('-save_periodic', help='Save actor, critic and memory periodically', action='store_true')
parser.add_argument('-seed', help='Random seed to be used', type=int, default=6)
parser.add_argument('-env', help='Environment Choices: (HalfCheetah-v2) (Ant-v2) (Reacher-v2) (Walker2d-v2) (Swimmer-v2) (Hopper-v2)', required=True)
parser.add_argument('-logdir', help='Folder where to save results', type=str, required=True)
class Parameters:

    def __init__(self, cla):
        # Number of Frames to Run
        cla = cla.parse_args()
        self.seed = cla.seed
        self.env_name = cla.env
        if cla.env == 'Hopper-v2':
            self.num_frames = 4000000
        elif cla.env == 'Ant-v2':
            self.num_frames = 6000000
        elif cla.env == 'Walker2d-v2':
            self.num_frames = 8000000
        else:
            self.num_frames = 2000000

        self.action_dim = None
        # Todo  动作维度和观察维度
        self.observation_dim = None

        # Num of trials meige 每个net跑几次，取平均值
        self.num_evals = 1
        if cla.env == 'Hopper-v2' or cla.env == 'Reacher-v2':
            self.num_evals = 5
        elif cla.env == 'Walker2d-v2':
            self.num_evals = 3
        else:
            self.num_evals = 1


        self.is_cuda = True
        self.is_memory_cuda = True

        self.gamma = 0.99  #折扣因子
        self.batch_size = 128
        self.buffer_size = 1000000


        self.pop_size = 10 #种群个数
        self.crossover_prob = 0.0 #杂交概率
        self.mutation_prob = 0.9 #突变概率

        self.individual_bs = 8000 #每个个体buffersize

        self.tau =0.001 #ddpg中的软更新
        self.ls = 128  # 隐藏层大小
        self.use_ln = True
        self.device = torch.device('cuda')
        #精英数目占比
        if cla.env == 'Reacher-v2' or cla.env == 'Walker2d-v2' or cla.env == 'Ant-v2' or cla.env == 'Hopper-v2':
            self.elite_fraction = 0.2
        else:
            self.elite_fraction = 0.1

            self.save_foldername = 'R_ERL/'
            if not os.path.exists(self.save_foldername): os.makedirs(self.save_foldername)
        #训练结果的文件夹
        self.save_foldername = cla.logdir
        if not os.path.exists(self.save_foldername):
            os.makedirs(self.save_foldername)



class Agent:
    def __init__(self, args, env):
        self.args = args
        self.env = env
        self.pop = []
        self.replay_buffer = repaly_memory.ReplayMemory(args.buffer_size)

        for _ in range(self.args.pop_size):
            self.pop.append(ddpg.GeneticAgent(args))


        # Turn off gradients and put in eval mode
        for ac in self.pop: ac.actor.eval()

        self.rl_agent = ddpg.DDPG(args)

        self.num_games = 0;
        self.num_frames = 0;
        self.iterations = 0;
        self.gen_frames = 0
#actor与环境交互
    def evaluate(self, agent: ddpg.GeneticAgent or ddpg.DDPG, store_transition=True):
        total_reward = 0.0
        state = self.env.reset()
        done = False

        while not done:
            if store_transition: self.num_frames += 1; self.gen_frames += 1
            action = agent.actor.select_action(np.array(state))
            #action.clamp(-1, 1)

            next_state, reward, done, info = self.env.step(action.flatten())
            if store_transition:
                self.replay_buffer.push(state, action, next_state, reward, done)
                #agent.buffer.push(state, action, next_state, reward, done)

            total_reward += reward
            state = next_state


        if store_transition: self.num_games += 1

        return total_reward

    def rl_to_evo(self, rl_net, evo_net):
        for target_param, param in zip(evo_net.parameters(), rl_net.parameters()):
            target_param.data.copy_(param.data)


    def train(self):

        all_fitness = []
        for agent in self.pop:
            fitness = 0
            for _ in range(self.args.num_evals): fitness += self.evaluate(agent)
            all_fitness.append(fitness/self.args.num_evals)
        print(all_fitness)


        best_train_fitness = max(all_fitness)
        worst_index = all_fitness.index(min(all_fitness))

        champ_index = all_fitness.index(max(all_fitness))

        test_score = 0.0
        for _ in range(5): test_score += self.evaluate(self.pop[champ_index], store_transition=False) / 5.0
        print(test_score)

        elite_index = []
        #Todo 选出精英片段的索引 elite_index


        ####################### DDPG #########################
        # DDPG Experience Collection
        self.evaluate(self.rl_agent)  # Train

        batch = self.replay_buffer.sample(self.args.batch_size)
        #batch = repaly_memory.Transition(*zip(*transitions))
        pgl, delta = self.rl_agent.update_parameters(batch)
        print("pgj:", pgl)
        self.rl_to_evo(self.rl_agent.actor, self.pop[worst_index].actor)
        #self.evolver.rl_policy = worst_index
        print('Synch from RL --> Nevo')



        return best_train_fitness, test_score, elite_index

if __name__ == "__main__":
    parameters = Parameters(parser)  # Create the Parameters class
    tracker = utils.Tracker(parameters, ['erl'], '_score.csv')  # Initiate tracker
    frame_tracker = utils.Tracker(parameters, ['frame_erl'], '_score.csv')  # Initiate tracker
    time_tracker = utils.Tracker(parameters, ['time_erl'], '_score.csv')

    # Create Env
    env = utils.NormalizedActions(gym.make(parameters.env_name))  # 因为action的输出是网络中【-1，1】的大小
    parameters.action_dim = env.action_space.shape[0]
    parameters.state_dim = env.observation_space.shape[0]


    # Seed
    env.seed(parameters.seed)
    torch.manual_seed(parameters.seed)
    np.random.seed(parameters.seed)
    random.seed(parameters.seed)

#Create Agent
    agent = Agent(parameters, env)
    print('Running', parameters.env_name, ' State_dim:', parameters.state_dim, ' Action_dim:', parameters.action_dim)
    next_save = 100
    time_start = time.time()
    while agent.num_frames <= parameters.num_frames:
        best_train_fitness, erl_score, elite_index = agent.train()
        print('#Games:', agent.num_games, '#Frames:', agent.num_frames, ' Epoch_Max:',
              '%.2f' % best_train_fitness if best_train_fitness != None else None, ' Test_Score:',
              '%.2f' % erl_score if erl_score != None else None, ' Avg:', '%.2f' % tracker.all_tracker[0][1],
              'ENV ' + parameters.env_name)
       #print('RL Selection Rate: Elite/Selected/Discarded',
        #      '%.2f' % (agent.evolver.selection_stats['elite'] / agent.evolver.selection_stats['total']),
        #      '%.2f' % (agent.evolver.selection_stats['selected'] / agent.evolver.selection_stats['total']),
        #      '%.2f' % (agent.evolver.selection_stats['discarded'] / agent.evolver.selection_stats['total']))
        ###*
        print()
        tracker.update([erl_score], agent.num_games)
        frame_tracker.update([erl_score], agent.num_frames)
        time_tracker.update([erl_score], time.time() - time_start)
        # Save Policy
        if agent.num_games > next_save:
            next_save += 100
            #if elite_index != None: torch.save(agent.pop[elite_index].state_dict(),
              #                                 parameters.save_foldername + 'evo_net')
            print("Progress Saved")