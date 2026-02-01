import numpy as np

class DynamicDESrestart:
    def __init__(self, budget=10000, dim=10, initial_popsize=None, F=0.5, CR=0.7, popsize_factor=2, stagnation_threshold=1e-6, entropy_threshold=0.1):
        self.budget = budget
        self.dim = dim
        self.initial_popsize = initial_popsize if initial_popsize is not None else 10 * self.dim
        self.popsize = self.initial_popsize
        self.F = F
        self.CR = CR
        self.popsize_factor = popsize_factor  # Factor to increase popsize
        self.stagnation_threshold = stagnation_threshold
        self.entropy_threshold = entropy_threshold
        self.eval_count = 0
        self.restart_count = 0


    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        self.population = np.random.uniform(lb, ub, size=(self.popsize, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.eval_count = self.popsize
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]

        while self.eval_count < self.budget:
            for i in range(self.popsize):
                # Mutation
                idxs = np.random.choice(self.popsize, 3, replace=False)
                x1, x2, x3 = self.population[idxs]
                mutant = x1 + self.F * (x2 - x3)
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
                    if self.fitness[i] < self.f_opt:
                        self.f_opt = self.fitness[i]
                        self.x_opt = self.population[i]

                if self.eval_count >= self.budget:
                    return self.f_opt, self.x_opt

            # Check for stagnation and restart or increase popsize
            if np.std(self.fitness) < self.stagnation_threshold or self.calculate_entropy(self.population) < self.entropy_threshold:
                self.restart_count += 1
                if self.popsize < self.initial_popsize * 4: # Limit popsize increase to a factor of 4
                    self.popsize = int(self.popsize * self.popsize_factor)
                    self.population = np.random.uniform(lb, ub, size=(self.popsize, self.dim))
                    self.fitness = np.array([func(x) for x in self.population])
                    self.eval_count += self.popsize
                    best_idx = np.argmin(self.fitness)
                    self.f_opt = self.fitness[best_idx]
                    self.x_opt = self.population[best_idx]

                else:  # Full Restart
                    self.population = np.random.uniform(lb, ub, size=(self.initial_popsize, self.dim))
                    self.fitness = np.array([func(x) for x in self.population])
                    self.popsize = self.initial_popsize
                    self.eval_count += self.initial_popsize
                    best_idx = np.argmin(self.fitness)
                    self.f_opt = self.fitness[best_idx]
                    self.x_opt = self.population[best_idx]
        return self.f_opt, self.x_opt

    def calculate_entropy(self, data, bins=10):
        """Calculates entropy of the population distribution in each dimension."""
        entropy = 0.0
        for i in range(self.dim):
            hist, _ = np.histogram(data[:, i], bins=bins, density=True)
            hist = hist + 1e-10  # Avoid log(0)
            pk = hist / np.sum(hist)
            entropy += -np.sum(pk * np.log2(pk))
        return entropy / self.dim