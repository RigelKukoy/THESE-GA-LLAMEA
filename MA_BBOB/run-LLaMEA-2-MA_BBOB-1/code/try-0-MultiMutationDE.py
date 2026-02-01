import numpy as np

class MultiMutationDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, mutation_strategies=None, selection_pressure=2):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.mutation_strategies = mutation_strategies if mutation_strategies is not None else [
            self.mutation_DE_rand1,
            self.mutation_DE_best1,
            self.mutation_DE_current_to_rand1,
            self.mutation_DE_current_to_best1,
        ]
        self.num_strategies = len(self.mutation_strategies)
        self.strategy_successes = np.ones(self.num_strategies)  # Initialize with ones to avoid division by zero
        self.selection_pressure = selection_pressure
        self.population = None
        self.fitness = None
        self.best_index = None
        self.f_opt = np.inf
        self.x_opt = None

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        self.best_index = np.argmin(self.fitness)
        self.f_opt = self.fitness[self.best_index]
        self.x_opt = self.population[self.best_index].copy()
        self.lb = func.bounds.lb
        self.ub = func.bounds.ub

    def mutation_DE_rand1(self, population, i, F):
        indices = np.random.choice(self.pop_size, 3, replace=False)
        x_r1, x_r2, x_r3 = population[indices]
        return population[i] + F * (x_r2 - x_r3)

    def mutation_DE_best1(self, population, i, F, best_index):
         indices = np.random.choice(self.pop_size, 2, replace=False)
         x_r1, x_r2 = population[indices]
         return population[i] + F * (population[best_index] - population[i]) + F * (x_r1 - x_r2)

    def mutation_DE_current_to_rand1(self, population, i, F):
        indices = np.random.choice(self.pop_size, 3, replace=False)
        x_r1, x_r2, x_r3 = population[indices]
        return population[i] + F * (x_r1 - population[i]) + F * (x_r2 - x_r3)

    def mutation_DE_current_to_best1(self, population, i, F, best_index):
        indices = np.random.choice(self.pop_size, 2, replace=False)
        x_r1, x_r2 = population[indices]
        return population[i] + F * (population[best_index] - population[i]) + F * (x_r1 - x_r2)
    

    def __call__(self, func):
        self.initialize_population(func)

        while self.budget > 0:
            strategy_probabilities = self.strategy_successes ** self.selection_pressure
            strategy_probabilities /= np.sum(strategy_probabilities)

            for i in range(self.pop_size):
                # Strategy selection
                strategy_index = np.random.choice(self.num_strategies, p=strategy_probabilities)
                mutation_strategy = self.mutation_strategies[strategy_index]

                # Mutation
                F = np.random.uniform(0.2, 0.8)
                try:
                    v = mutation_strategy(self.population, i, F, self.best_index)
                except TypeError:
                    v = mutation_strategy(self.population, i, F)

                v = np.clip(v, self.lb, self.ub)

                # Crossover
                CR = np.random.uniform(0.3, 0.9)
                j_rand = np.random.randint(self.dim)
                u = self.population[i].copy()
                for j in range(self.dim):
                    if np.random.rand() < CR or j == j_rand:
                        u[j] = v[j]

                # Evaluation
                f_u = func(u)
                self.budget -= 1

                # Selection
                if f_u < self.fitness[i]:
                    self.strategy_successes[strategy_index] += 1
                    self.fitness[i] = f_u
                    self.population[i] = u
                    if f_u < self.f_opt:
                        self.f_opt = f_u
                        self.x_opt = u.copy()
                        self.best_index = np.argmin(self.fitness)
                else:
                     self.strategy_successes[strategy_index] *= 0.9  # Reduce success if the strategy didn't improve

        return self.f_opt, self.x_opt