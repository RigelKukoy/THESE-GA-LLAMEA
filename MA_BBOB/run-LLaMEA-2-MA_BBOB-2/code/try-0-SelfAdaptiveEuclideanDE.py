import numpy as np

class SelfAdaptiveEuclideanDE:
    def __init__(self, budget=10000, dim=10, popsize=None, F_range=(0.1, 0.9), CR_range=(0.1, 0.9), neighborhood_size=5):
        self.budget = budget
        self.dim = dim
        self.popsize = popsize if popsize is not None else 10 * self.dim
        self.F_range = F_range
        self.CR_range = CR_range
        self.neighborhood_size = neighborhood_size
        self.population = None
        self.fitness = None
        self.F = None
        self.CR = None
        self.x_opt = None
        self.f_opt = np.inf
        self.eval_count = 0

    def initialize_population(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        self.population = np.random.uniform(lb, ub, size=(self.popsize, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.eval_count += self.popsize
        self.F = np.random.uniform(self.F_range[0], self.F_range[1], size=self.popsize)
        self.CR = np.random.uniform(self.CR_range[0], self.CR_range[1], size=self.popsize)
        self.x_opt = self.population[np.argmin(self.fitness)]
        self.f_opt = np.min(self.fitness)

    def euclidean_distance(self, x, y):
        return np.linalg.norm(x - y)

    def adapt_parameters(self):
        for i in range(self.popsize):
            # Find the neighborhood based on Euclidean distance in the search space
            distances = [self.euclidean_distance(self.population[i], self.population[j]) for j in range(self.popsize)]
            neighborhood_indices = np.argsort(distances)[:self.neighborhood_size]

            # Calculate the mean F and CR values from the neighborhood
            self.F[i] = np.mean(self.F[neighborhood_indices])
            self.CR[i] = np.mean(self.CR[neighborhood_indices])

            # Apply bounds to F and CR
            self.F[i] = np.clip(self.F[i], self.F_range[0], self.F_range[1])
            self.CR[i] = np.clip(self.CR[i], self.CR_range[0], self.CR_range[1])

    def __call__(self, func):
        self.initialize_population(func)
        lb = func.bounds.lb
        ub = func.bounds.ub

        while self.eval_count < self.budget:
            # Elitism: Keep the best individual
            elite_index = np.argmin(self.fitness)
            elite = self.population[elite_index].copy()
            elite_fitness = self.fitness[elite_index]

            for i in range(self.popsize):
                # Mutation
                donor_indices = np.random.choice(self.popsize, 3, replace=False)
                mutant = self.population[donor_indices[0]] + self.F[i] * (self.population[donor_indices[1]] - self.population[donor_indices[2]])
                mutant = np.clip(mutant, lb, ub)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.CR[i]
                trial = np.where(crossover_mask, mutant, self.population[i])

                # Selection
                f_trial = func(trial)
                self.eval_count += 1

                if f_trial < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = f_trial

                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

            # Parameter Adaptation
            self.adapt_parameters()

            # Elitism: Replace a random individual with the elite from the previous generation
            random_index = np.random.randint(self.popsize)
            self.population[random_index] = elite
            self.fitness[random_index] = elite_fitness

            if self.eval_count >= self.budget:
                break

        return self.f_opt, self.x_opt