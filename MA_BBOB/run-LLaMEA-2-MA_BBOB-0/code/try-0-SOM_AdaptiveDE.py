import numpy as np

class SOM_AdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, som_grid_size=10, initial_mutation_factor=0.5, initial_crossover_rate=0.7, learning_rate=0.1, sigma=1.0):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.som_grid_size = som_grid_size
        self.mutation_factor = initial_mutation_factor
        self.crossover_rate = initial_crossover_rate
        self.learning_rate = learning_rate
        self.sigma = sigma  # Neighborhood radius for SOM update
        self.population = None
        self.fitness = None
        self.som = np.random.rand(som_grid_size, som_grid_size, dim)  # Initialize SOM weights
        self.som_fitness = np.zeros((som_grid_size, som_grid_size))  # Fitness associated with each SOM node
        self.x_opt = None
        self.f_opt = np.inf
        self.success_mutation_factors = []
        self.success_crossover_rates = []

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, size=(self.pop_size, self.dim))

    def find_best_matching_unit(self, x):
        distances = np.sum((self.som - x)**2, axis=2)
        return np.unravel_index(np.argmin(distances), distances.shape)

    def update_som(self, x, fitness, bmu):
        # Gaussian neighborhood function
        for i in range(self.som_grid_size):
            for j in range(self.som_grid_size):
                distance = np.sqrt((i - bmu[0])**2 + (j - bmu[1])**2)
                influence = np.exp(-distance**2 / (2 * self.sigma**2))
                self.som[i, j] += self.learning_rate * influence * (x - self.som[i, j])
                self.som_fitness[i, j] = 0.9 * self.som_fitness[i,j] + 0.1 * fitness # EWMA fitness update for the node

    def differential_evolution(self, func, lb, ub):
        new_population = np.copy(self.population)
        new_fitness = np.copy(self.fitness)

        for i in range(self.pop_size):
            idxs = np.random.choice(self.pop_size, 3, replace=False)
            x_r1, x_r2, x_r3 = self.population[idxs]

            bmu = self.find_best_matching_unit(self.population[i])
            x_som = self.som[bmu] # Use the best matching unit from SOM.
            
            mutant = x_r1 + self.mutation_factor * (x_r2 - x_r3) + self.mutation_factor * (x_som - self.population[i]) # Use SOM for guiding mutation

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
                self.update_som(trial, f_trial, bmu)

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

        # Initial SOM training with initial population
        for x, fitness in zip(self.population, self.fitness):
             bmu = self.find_best_matching_unit(x)
             self.update_som(x, fitness, bmu)


        while self.budget > 0:
            self.differential_evolution(func, lb, ub)
            self.adapt_parameters()

        return self.f_opt, self.x_opt