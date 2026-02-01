import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.5, CR=0.9, adaptation_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.adaptation_rate = adaptation_rate
        self.success_history_F = []
        self.success_history_CR = []
        self.memory_size = 10 

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size

        for i in range(self.pop_size):
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = self.population[i]

        while self.budget > 0:
            successful_F = []
            successful_CR = []

            for i in range(self.pop_size):
                # Mutation
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                x_r1, x_r2, x_r3 = self.population[idxs]
                mutant = x_r1 + self.F * (x_r2 - x_r3)
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                # Crossover
                crossover = np.random.uniform(size=self.dim) < self.CR
                trial = np.where(crossover, mutant, self.population[i])
                trial = np.clip(trial, func.bounds.lb, func.bounds.ub)

                # Selection
                f_trial = func(trial)
                self.budget -= 1

                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial

                if f_trial < fitness[i]:
                    successful_F.append(self.F)
                    successful_CR.append(self.CR)
                    fitness[i] = f_trial
                    self.population[i] = trial
            
            # Adapt parameters
            if successful_F:
                self.F = np.mean(successful_F)
            else:
                self.F = 0.5  # Reset if no success

            if successful_CR:
                self.CR = np.mean(successful_CR)
            else:
                self.CR = 0.9 # Reset if no success

            # Apply Adaptation Rate
            self.F = self.F * (1 - self.adaptation_rate) + np.random.normal(0.5, 0.1) * self.adaptation_rate
            self.CR = self.CR * (1 - self.adaptation_rate) + np.random.normal(0.9, 0.1) * self.adaptation_rate
            self.F = np.clip(self.F, 0.1, 0.9)
            self.CR = np.clip(self.CR, 0.1, 1.0)

        return self.f_opt, self.x_opt