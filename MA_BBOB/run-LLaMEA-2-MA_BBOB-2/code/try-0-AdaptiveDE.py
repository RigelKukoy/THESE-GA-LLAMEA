import numpy as np

class AdaptiveDE:
    def __init__(self, budget=10000, dim=10, initial_popsize=None, F=0.5, CR=0.7, stagnation_threshold=1e-6, popsize_reduction_factor=0.9, popsize_increase_factor=1.1, min_popsize=10, cauchy_scale=0.1, gaussian_scale=0.1):
        self.budget = budget
        self.dim = dim
        self.initial_popsize = initial_popsize if initial_popsize is not None else 10 * self.dim
        self.popsize = self.initial_popsize
        self.F = F
        self.CR = CR
        self.stagnation_threshold = stagnation_threshold
        self.popsize_reduction_factor = popsize_reduction_factor
        self.popsize_increase_factor = popsize_increase_factor
        self.min_popsize = min_popsize
        self.cauchy_scale = cauchy_scale
        self.gaussian_scale = gaussian_scale

        self.best_fitness_history = []

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        self.population = np.random.uniform(lb, ub, size=(self.popsize, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.eval_count = self.popsize
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]
        self.best_fitness_history.append(self.f_opt)

        stagnation_counter = 0
        mutation_strength = 1.0  # Initial mutation strength

        while self.eval_count < self.budget:
            for i in range(self.popsize):
                # Mutation: Combined Cauchy and Gaussian
                idxs = np.random.choice(self.popsize, 3, replace=False)
                x1, x2, x3 = self.population[idxs]

                cauchy_component = np.random.standard_cauchy(size=self.dim) * self.cauchy_scale
                gaussian_component = np.random.normal(0, 1, size=self.dim) * self.gaussian_scale

                mutant = x1 + self.F * mutation_strength * (cauchy_component + gaussian_component) * (x2 - x3)
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

            # Adaptive Population Size
            if len(self.best_fitness_history) > 1:
                if abs(self.best_fitness_history[-1] - self.best_fitness_history[-2]) < self.stagnation_threshold:
                    stagnation_counter += 1
                else:
                    stagnation_counter = 0

            if stagnation_counter > 5:  # Stagnation detected
                # Reduce population size
                new_popsize = int(self.popsize * self.popsize_reduction_factor)
                new_popsize = max(new_popsize, self.min_popsize)

                if new_popsize < self.popsize:
                    self.popsize = new_popsize
                    self.population = self.population[np.argsort(self.fitness)[:self.popsize]]
                    self.fitness = self.fitness[np.argsort(self.fitness)[:self.popsize]]
                    stagnation_counter = 0
                    mutation_strength *= 0.8 # Decrease mutation strength upon stagnation
                else:
                     mutation_strength *= 1.2 # Increase mutation strength when not reducing popsize to escape local optima.

            elif self.eval_count/self.budget > 0.75 and self.popsize < self.initial_popsize * 2: #Late increase to improve the optimum
                new_popsize = int(self.popsize * self.popsize_increase_factor)
                if new_popsize > self.popsize and self.eval_count + new_popsize - self.popsize < self.budget:
                     new_individuals = np.random.uniform(lb, ub, size=(new_popsize - self.popsize, self.dim))
                     new_fitness = np.array([func(x) for x in new_individuals])
                     self.eval_count += new_popsize - self.popsize
                     self.population = np.vstack((self.population, new_individuals))
                     self.fitness = np.concatenate((self.fitness, new_fitness))
                     self.popsize = new_popsize


            self.best_fitness_history.append(self.f_opt)

        return self.f_opt, self.x_opt