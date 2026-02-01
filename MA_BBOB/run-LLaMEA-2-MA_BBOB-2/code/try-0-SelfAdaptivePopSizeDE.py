import numpy as np

class SelfAdaptivePopSizeDE:
    def __init__(self, budget=10000, dim=10, popsize_init=None, F=0.5, CR=0.7, popsize_min=4, popsize_max=200):
        self.budget = budget
        self.dim = dim
        self.popsize = popsize_init if popsize_init is not None else 10 * self.dim
        self.popsize = int(self.popsize)
        self.F = F
        self.CR = CR
        self.popsize_min = popsize_min
        self.popsize_max = popsize_max
        self.success_rate = 0.5
        self.success_history = []

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        self.population = np.random.uniform(lb, ub, size=(self.popsize, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.eval_count = self.popsize
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]

        while self.eval_count < self.budget:
            successful_mutations = 0
            for i in range(self.popsize):
                # Mutation
                idxs = np.random.choice(self.popsize, 3, replace=False)
                x_r1, x_r2, x_r3 = self.population[idxs]
                mutant = self.population[i] + self.F * (x_r1 - x_r2)

                mutant = np.clip(mutant, lb, ub)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, self.population[i])

                # Selection
                f_trial = func(trial)
                self.eval_count += 1

                if f_trial < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = f_trial
                    successful_mutations += 1
                    if self.fitness[i] < self.f_opt:
                        self.f_opt = self.fitness[i]
                        self.x_opt = self.population[i]

            # Adjust population size based on success rate
            success_rate = successful_mutations / self.popsize
            self.success_history.append(success_rate)
            if len(self.success_history) > 10:
                self.success_history = self.success_history[-10:]
            self.success_rate = np.mean(self.success_history)
            
            if self.success_rate > 0.3:
                self.popsize = min(self.popsize + 1, self.popsize_max)
            elif self.success_rate < 0.1:
                self.popsize = max(self.popsize - 1, self.popsize_min)
            
            self.popsize = int(self.popsize)

            # Resize population
            if self.popsize != self.population.shape[0]:
                if self.popsize > self.population.shape[0]:
                    # Add new individuals randomly
                    new_individuals = np.random.uniform(lb, ub, size=(self.popsize - self.population.shape[0], self.dim))
                    self.population = np.vstack((self.population, new_individuals))
                    new_fitness = np.array([func(x) for x in new_individuals])
                    self.fitness = np.concatenate((self.fitness, new_fitness))
                    self.eval_count += new_individuals.shape[0]
                else:
                    # Remove worst individuals
                    indices_to_keep = np.argsort(self.fitness)[:self.popsize]
                    self.population = self.population[indices_to_keep]
                    self.fitness = self.fitness[indices_to_keep]

        return self.f_opt, self.x_opt