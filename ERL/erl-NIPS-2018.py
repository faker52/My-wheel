import argparse
from core import ddpg
import torch
Environment_Name = 'Swimmer-v2'



class Parameters:

    def __init__(self, env_name):
        # Number of Frames to Run
        if env_name == 'Hopper-v2':
            self.num_frames = 4000000
        elif env_name == 'Ant-v2':
            self.num_frames = 6000000
        elif env_name == 'Walker2d-v2':
            self.num_frames = 8000000
        else:
            self.num_frames = 2000000

        self.action_dim = None
        # Todo  动作维度和观察维度
        self.observation_dim = None

        # Num of trials
        self.num_evals = 1
        if env_name == 'Hopper-v2' or env_name == 'Reacher-v2':
            self.num_evals = 5
        elif env_name == 'Walker2d-v2':
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



class Agent:
    def __init__(self, args, env):
        self.args = args
        self.env = env
        self.pop = []
        for _ in range(self.args.pop_size):
            self.pop.append(ddpg.Actor(args))
        self.replay_buffer = []

        # Turn off gradients and put in eval mode
        for actor in self.pop: actor.eval()

        self.rl_agent = ddpg.DDPG(args)





#actor与环境交互
    def evaluate(self, net, store_transition=True):
        total_reward = 0.0
        state = self.env.reset()

        state = torch.FloatTensor(state).cuda()
        done = False

        while not done:
            if store_transition: self.num_frames += 1; self.gen_frames += 1

            action = net.forward(state)
            #action.clamp(-1, 1)
            next_state, reward, done, info = self.env.step(action.cpu().data.numpy())
            if store_transition: self.replay_buffer.push(state, action, next_state, reward, done)

            next_state = torch.FloatTensor(next_state).cuda()
            total_reward += reward
            state = next_state


        if store_transition: self.num_games += 1

        return total_reward


    def train(self):

        all_fitness = []
        for net in self.pop:
            fitness = 0
            for _ in range(self.args.num_evals): fitness += self.evaluate(net)
            all_fitness.append(fitness/self.args.num_evals)

        best_train_fitness = max(all_fitness)
        worst_index = all_fitness.index(min(all_fitness))

        champ_index = all_fitness.index(max(all_fitness))
        test_score = 0.0
        for _ in range(5): test_score += self.evaluate(self.pop[champ_index], store_transition=False) / 5.0


