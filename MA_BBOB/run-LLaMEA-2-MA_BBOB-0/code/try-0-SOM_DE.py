import numpy as np

class SOM_DE:
    def __init__(self, budget=10000, dim=10, pop_size=40, som_grid_size=10, learning_rate=0.1, mutation_factor=0.5, crossover_rate=0.7):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.som_grid_size = som_grid_size
        self.learning_rate = learning_rate
        self.mutation_factor = mutation_factor
        self.crossover_rate = crossover_rate
        self.som = None
        self.population = None
        self.fitness = None
        self.x_opt = None
        self.f_opt = np.inf

    def initialize_som(self):
        self.som = np.random.uniform(-1, 1, size=(self.som_grid_size, self.som_grid_size, self.dim))

    def find_best_matching_unit(self, x):
        distances = np.sum((self.som - x)**2, axis=2)
        bmu_index = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu_index

    def update_som(self, x, bmu_index):
        distance = np.sqrt((np.arange(self.som_grid_size) - bmu_index[0])**2[:, None] + (np.arange(self.som_grid_size) - bmu_index[1])**2[None, :])
        influence = np.exp(-(distance**2) / (2 * (self.som_grid_size/4)**2))  # Gaussian neighborhood
        
        for i in range(self.som_grid_size):
            for j in range(self.som_grid_size):
                self.som[i, j] += self.learning_rate * influence[i, j] * (x - self.som[i, j])


    def differential_evolution(self, func):
        for i in range(self.pop_size):
            # Mutation
            indices = np.random.choice(self.pop_size, 3, replace=False)
            x_r1, x_r2, x_r3 = self.population[indices]
            x_mutated = x_r1 + self.mutation_factor * (x_r2 - x_r3)
            x_mutated = np.clip(x_mutated, func.bounds.lb, func.bounds.ub)

            # Crossover
            x_trial = np.copy(self.population[i])
            for j in range(self.dim):
                if np.random.rand() < self.crossover_rate:
                    x_trial[j] = x_mutated[j]

            f_trial = func(x_trial)
            self.budget -= 1

            if f_trial < self.fitness[i]:
                self.fitness[i] = f_trial
                self.population[i] = x_trial

                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = x_trial
            
            if self.budget <= 0:
                break


    def __call__(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size

        best_index = np.argmin(self.fitness)
        self.x_opt = self.population[best_index]
        self.f_opt = self.fitness[best_index]

        self.initialize_som()

        while self.budget > 0:
            # SOM Training
            for x in self.population:
                bmu_index = self.find_best_matching_unit(x)
                self.update_som(x, bmu_index)

            # Differential Evolution
            self.differential_evolution(func)

            best_index = np.argmin(self.fitness)
            if self.fitness[best_index] < self.f_opt:
                self.f_opt = self.fitness[best_index]
                self.x_opt = self.population[best_index]
            
            self.learning_rate = 0.1 * (self.budget / 10000)


        return self.f_opt, self.x_opt