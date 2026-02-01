import numpy as np

class AdaptiveArchiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, archive_size=100, initial_mutation_factor=0.5, initial_crossover_rate=0.7):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.archive_size = archive_size
        self.mutation_factor = initial_mutation_factor
        self.crossover_rate = initial_crossover_rate
        self.population = None
        self.fitness = None
        self.archive = []
        self.archive_fitness = []
        self.x_opt = None
        self.f_opt = np.inf
        self.success_mutation_factors = []
        self.success_crossover_rates = []

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, size=(self.pop_size, self.dim))

    def update_archive(self, x, f):
        if len(self.archive) < self.archive_size:
            self.archive.append(x)
            self.archive_fitness.append(f)
        else:
            # Replace worst element in archive if new solution is better
            worst_index = np.argmax(self.archive_fitness)
            if f < self.archive_fitness[worst_index]:
                self.archive[worst_index] = x
                self.archive_fitness[worst_index] = f

    def differential_evolution(self, func, lb, ub):
        new_population = np.copy(self.population)
        new_fitness = np.copy(self.fitness)

        for i in range(self.pop_size):
            idxs = np.random.choice(self.pop_size, 3, replace=False)
            x_r1, x_r2, x_r3 = self.population[idxs]

            # Incorporate archive member into mutation
            if len(self.archive) > 0:
                x_archive = self.archive[np.random.randint(0, len(self.archive))]
                mutant = x_r1 + self.mutation_factor * (x_r2 - x_r3) + self.mutation_factor * (x_archive - self.population[i]) # ADDED ARCHIVE
            else:
                mutant = x_r1 + self.mutation_factor * (x_r2 - x_r3)

            mutant = np.clip(mutant, lb, ub)

            crossover = np.random.uniform(size=self.dim) < self.crossover_rate
            trial = np.where(crossover, mutant, self.population[i])

            f_trial = func(trial)
            self.budget -= 1

            if f_trial < self.f_opt:
                self.f_opt = f_trial
                self.x_opt = trial

            if f_trial < self.fitness[i]:
                new_fitness[i] = f_trial
                new_population[i] = trial
                self.update_archive(trial, f_trial)

                # Store successful parameters
                self.success_mutation_factors.append(self.mutation_factor)
                self.success_crossover_rates.append(self.crossover_rate)

        self.population = new_population
        self.fitness = new_fitness

    def adapt_parameters(self):
        # Adapt mutation factor and crossover rate based on success history
        if len(self.success_mutation_factors) > 0:
            self.mutation_factor = np.mean(self.success_mutation_factors)
            self.crossover_rate = np.mean(self.success_crossover_rates)

            # Add some noise to prevent stagnation
            self.mutation_factor = np.clip(self.mutation_factor + np.random.normal(0, 0.1), 0.1, 1.0)
            self.crossover_rate = np.clip(self.crossover_rate + np.random.normal(0, 0.1), 0.1, 1.0)

            self.success_mutation_factors = [] #reset
            self.success_crossover_rates = [] #reset
        else:
            # If no success, increase exploration
            self.mutation_factor = np.clip(self.mutation_factor + 0.1, 0.1, 1.0)
            self.crossover_rate = np.clip(self.crossover_rate - 0.1, 0.1, 1.0) # Reduce to exploit less.

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size

        best_index = np.argmin(self.fitness)
        self.x_opt = self.population[best_index]
        self.f_opt = self.fitness[best_index]
        self.update_archive(self.x_opt, self.f_opt)

        while self.budget > 0:
            self.differential_evolution(func, lb, ub)
            self.adapt_parameters()

        return self.f_opt, self.x_opt