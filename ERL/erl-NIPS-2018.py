import argparse

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
            self.pop.append()
            # Todo : ddpg 中的 actor
        self.replay_buffer = []


