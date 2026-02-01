import numpy as np

class WaveletAdaptiveDE:
    def __init__(self, budget=10000, dim=10, popsize=None, F_initial=0.5, CR_initial=0.7, restart_threshold=100):
        self.budget = budget
        self.dim = dim
        self.popsize = popsize if popsize is not None else 10 * self.dim
        self.F = F_initial
        self.CR = CR_initial
        self.restart_threshold = restart_threshold
        self.best_fitness_history = []

    def wavelet_mutation(self, x, level=3):
        mutant = x.copy()
        for i in range(self.dim):
            # Apply Discrete Wavelet Transform (Haar wavelet)
            coeff = x[i]
            for _ in range(level):
                detail = np.random.normal(0, 0.1)  # Add wavelet detail coefficient
                coeff += detail
            mutant[i] = coeff
        return mutant

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

        while self.eval_count < self.budget:
            for i in range(self.popsize):
                # Self-adaptive parameters
                self.F = np.clip(np.random.normal(self.F, 0.1), 0.1, 1.0)
                self.CR = np.clip(np.random.normal(self.CR, 0.1), 0.1, 1.0)

                # Mutation
                idxs = np.random.choice(self.popsize, 3, replace=False)
                x1, x2, x3 = self.population[idxs]
                mutant = x1 + self.F * (x2 - x3)
                mutant = np.clip(mutant, lb, ub)

                # Wavelet mutation
                mutant = self.wavelet_mutation(mutant)
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

            # Stagnation Check and Restart
            self.best_fitness_history.append(self.f_opt)
            if len(self.best_fitness_history) > self.restart_threshold:
                if abs(self.best_fitness_history[-1] - self.best_fitness_history[-self.restart_threshold]) < 1e-6:
                    stagnation_counter += 1
                else:
                    stagnation_counter = 0

                if stagnation_counter > self.restart_threshold // 2:
                    # Restart Population
                    self.population = np.random.uniform(lb, ub, size=(self.popsize, self.dim))
                    self.fitness = np.array([func(x) for x in self.population])
                    self.eval_count += self.popsize
                    self.f_opt = np.min(self.fitness)
                    self.x_opt = self.population[np.argmin(self.fitness)]
                    self.best_fitness_history = [self.f_opt] # Reset fitness history
                    stagnation_counter = 0 # Reset stagnation counter

        return self.f_opt, self.x_opt