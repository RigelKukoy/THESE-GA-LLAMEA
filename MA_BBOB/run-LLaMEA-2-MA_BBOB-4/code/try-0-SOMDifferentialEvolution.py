import numpy as np
from minisom import MiniSom

class SOMDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=40, som_grid_size=5, initial_Cr=0.5, initial_F=0.7, learning_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.som_grid_size = som_grid_size
        self.Cr = np.full(pop_size, initial_Cr)  # Individual crossover rates
        self.F = np.full(pop_size, initial_F)  # Individual mutation factors
        self.learning_rate = learning_rate
        self.population = None
        self.fitness = None
        self.f_opt = np.Inf
        self.x_opt = None
        self.generation = 0
        self.som = None
        self.cluster_labels = None
        self.min_F = 0.1
        self.max_F = 1.0

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]

    def initialize_som(self):
        self.som = MiniSom(self.som_grid_size, self.som_grid_size, self.dim, sigma=0.3, learning_rate=0.5)
        self.som.random_weights_init(self.population)
        self.som.train_random(self.population, 100)  # Train SOM for a few iterations

    def assign_clusters(self):
         self.cluster_labels = [self.som.winner(x) for x in self.population]

    def evolve(self, func):
        new_population = np.copy(self.population)
        new_fitness = np.copy(self.fitness)

        for i in range(self.pop_size):
            # Get cluster label for the individual
            cluster = self.cluster_labels[i]
            
            # Mutation strategy based on cluster (example: different F values)
            if cluster[0] % 2 == 0:  # Example: even rows in SOM grid
                F = self.F[i]  # Use individual F
                mutation_strategy = 1
            else:  # Example: odd rows in SOM grid
                F = np.random.uniform(0.5, 1.0) # Use different F
                mutation_strategy = 2

            # Mutation
            if mutation_strategy == 1:
                r1, r2, r3 = np.random.choice(self.pop_size, 3, replace=False)
                mutant = self.population[r1] + F * (self.population[r2] - self.population[r3])
            elif mutation_strategy == 2:
                # Using current best individual
                best_idx = np.argmin(self.fitness)
                r1, r2 = np.random.choice(self.pop_size, 2, replace=False)
                mutant = self.population[i] + F * (self.population[best_idx] - self.population[i]) + F * (self.population[r1] - self.population[r2])
            
            # Crossover
            for j in range(self.dim):
                if np.random.rand() < self.Cr[i]:
                    new_population[i, j] = mutant[j]
                else:
                    new_population[i, j] = self.population[i, j]

            new_population[i] = np.clip(new_population[i], func.bounds.lb, func.bounds.ub)

            # Evaluation
            new_fitness[i] = func(new_population[i])
            self.budget -= 1

            # Selection
            if new_fitness[i] < self.fitness[i]:
                self.population[i] = new_population[i]
                self.fitness[i] = new_fitness[i]

                if new_fitness[i] < self.f_opt:
                    self.f_opt = new_fitness[i]
                    self.x_opt = new_population[i]

        # Adapt SOM
        self.som.train_random(self.population, 10)
        self.assign_clusters()

    def __call__(self, func):
        self.initialize_population(func)
        self.initialize_som()
        self.assign_clusters()

        while self.budget > self.pop_size:
            self.evolve(func)
            self.generation += 1
        return self.f_opt, self.x_opt