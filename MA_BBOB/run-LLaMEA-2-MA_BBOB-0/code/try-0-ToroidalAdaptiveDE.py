import numpy as np

class ToroidalAdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, neighborhood_size=5):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.neighborhood_size = neighborhood_size
        self.F = 0.5 * np.ones(pop_size)
        self.CR = 0.9 * np.ones(pop_size)
        self.population = None
        self.fitness = None
        self.best_fitness = np.inf
        self.best_solution = None
        self.success_F = [[] for _ in range(pop_size)]
        self.success_CR = [[] for _ in range(pop_size)]
        self.epsilon = 1e-6

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size

        best_index = np.argmin(self.fitness)
        self.best_fitness = self.fitness[best_index]
        self.best_solution = self.population[best_index].copy()

    def toroidal_neighbors(self, index):
        neighbors = []
        for i in range(-self.neighborhood_size // 2, self.neighborhood_size // 2 + 1):
            neighbor_index = (index + i) % self.pop_size
            neighbors.append(neighbor_index)
        return neighbors

    def evolve(self, func):
        for i in range(self.pop_size):
            neighbors = self.toroidal_neighbors(i)
            neighbor_population = self.population[neighbors]
            neighbor_fitness = self.fitness[neighbors]

            # Mutation
            idxs = np.random.choice(len(neighbors), 3, replace=False)
            x_r1, x_r2, x_r3 = neighbor_population[idxs]
            mutant = self.population[i] + self.F[i] * (x_r1 - x_r2)

            mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

            # Crossover
            crossover_mask = np.random.rand(self.dim) < self.CR[i]
            trial = np.where(crossover_mask, mutant, self.population[i])

            # Evaluation
            trial_fitness = func(trial)
            self.budget -= 1

            if trial_fitness < self.best_fitness:
                self.best_fitness = trial_fitness
                self.best_solution = trial.copy()
            
            # Selection and Parameter Adaptation
            if trial_fitness < self.fitness[i]:
                self.success_F[i].append(self.F[i])
                self.success_CR[i].append(self.CR[i])
                
                self.population[i] = trial
                self.fitness[i] = trial_fitness
            
            # Update F and CR based on neighborhood performance
            if self.success_F[i]:
                self.F[i] = np.mean(self.success_F[i])
                self.CR[i] = np.mean(self.success_CR[i])
            
            self.F[i] = np.clip(self.F[i], 0.1, 1.0)
            self.CR[i] = np.clip(self.CR[i], 0.1, 1.0)

    def __call__(self, func):
        self.initialize_population(func)

        while self.budget > 0:
            self.evolve(func)

        return self.best_fitness, self.best_solution