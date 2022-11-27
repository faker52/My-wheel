
import numpy as np
import fastrand, math

class SSNE:
    def __init__(self, args, critic, evaluate):
        self.num_elitists = int(self.args.elite_fraction * args.pop_size)


    def epoch(self, pop, all_fitness):
        index_rank = np.argsort(all_fitness)[::-1]  # 对所有reward进行降序排序，输出的是对应的index
        elitist_index = index_rank[:self.num_elitists]  # Elitist indexes safeguard num_elitists = 比例×pop.size，选出精英的索引（index）

        offsprings = self.selection_tournament(index_rank, num_offsprings=len(index_rank) - self.num_elitists,
                                           tournament_size=3)




    def selection_tournament(self, index_rank, num_offsprings, tournament_size):
        total_choices = len(index_rank)
        offsprings = []
        for i in range(num_offsprings):
            winner = np.min(np.random.randint(total_choices, size=tournament_size))
            offsprings.append(index_rank[winner])

        offsprings = list(set(offsprings))  # Find unique offsprings
        if len(offsprings) % 2 != 0:  # Number of offsprings should be even
            offsprings.append(offsprings[fastrand.pcg32bounded(len(offsprings))])
        return offsprings
