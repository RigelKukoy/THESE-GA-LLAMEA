import numpy as np

class MirroredAdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, memory_size=10):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.memory_size = memory_size
        self.F = 0.5
        self.CR = 0.9
        self.F_memory = np.ones(memory_size) * 0.5
        self.CR_memory = np.ones(memory_size) * 0.9
        self.memory_idx = 0
        self.success_F = []
        self.success_CR = []

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        lb = func.bounds.lb
        ub = func.bounds.ub
        self.population = np.random.uniform(lb, ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size

        for i in range(self.pop_size):
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = self.population[i]

        while self.budget > 0:
            for i in range(self.pop_size):
                # Parameter adaptation
                self.F = self.F_memory[self.memory_idx]
                self.CR = self.CR_memory[self.memory_idx]

                # Mutation
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                x_1, x_2, x_3 = self.population[idxs]
                mutant = x_1 + self.F * (x_2 - x_3)

                # Mirrored sampling to handle boundary violations
                for j in range(self.dim):
                    if mutant[j] < lb:
                        mutant[j] = lb + np.abs(mutant[j] - lb)
                    if mutant[j] > ub:
                        mutant[j] = ub - np.abs(mutant[j] - ub)
                        
                # Crossover
                crossover = np.random.uniform(size=self.dim) < self.CR
                trial = np.where(crossover, mutant, self.population[i])

                # Selection
                f_trial = func(trial)
                self.budget -= 1

                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial

                if f_trial < fitness[i]:
                    self.success_F.append(self.F)
                    self.success_CR.append(self.CR)
                    fitness[i] = f_trial
                    self.population[i] = trial
            
            # Update memory
            if self.success_F:
                self.F_memory[self.memory_idx] = np.mean(self.success_F)
                self.CR_memory[self.memory_idx] = np.mean(self.success_CR)
            self.success_F = []
            self.success_CR = []
            self.memory_idx = (self.memory_idx + 1) % self.memory_size

        return self.f_opt, self.x_opt