import numpy as np

class SOMDE:
    def __init__(self, budget=10000, dim=10, pop_size=20, som_grid_size=5, de_mutation_factor=0.5, de_crossover_rate=0.7):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.som_grid_size = som_grid_size  # Size of the SOM grid (som_grid_size x som_grid_size)
        self.de_mutation_factor = de_mutation_factor
        self.de_crossover_rate = de_crossover_rate
        self.som = None
        self.learning_rate = 0.1
        self.neighborhood_radius = som_grid_size // 2

    def initialize_som(self):
        """Initializes the Self-Organizing Map (SOM)."""
        self.som = np.random.uniform(-1, 1, size=(self.som_grid_size, self.som_grid_size, self.dim))

    def find_best_matching_unit(self, x):
        """Finds the best matching unit (BMU) in the SOM for a given input vector x."""
        distances = np.sum((self.som - x)**2, axis=2)  # Euclidean distance
        bmu_index = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu_index

    def update_som(self, x, bmu_index):
        """Updates the SOM weights based on the input vector x and the BMU."""
        for i in range(self.som_grid_size):
            for j in range(self.som_grid_size):
                distance = np.sqrt((i - bmu_index[0])**2 + (j - bmu_index[1])**2)
                if distance <= self.neighborhood_radius:
                    influence = np.exp(-distance**2 / (2 * self.neighborhood_radius**2))
                    self.som[i, j] += self.learning_rate * influence * (x - self.som[i, j])

    def differential_evolution(self, func):
        """Applies Differential Evolution to the population."""
        for i in range(self.pop_size):
            # Choose three random individuals (excluding the current one)
            indices = np.random.choice(self.pop_size, 3, replace=False)
            if i in indices:
                indices = np.random.choice(self.pop_size, 3, replace=False)
                while i in indices:
                    indices = np.random.choice(self.pop_size, 3, replace=False)
                    
            x_r1, x_r2, x_r3 = self.population[indices]

            # Mutation
            x_mutated = x_r1 + self.de_mutation_factor * (x_r2 - x_r3)
            x_mutated = np.clip(x_mutated, func.bounds.lb, func.bounds.ub)

            # Crossover
            x_trial = np.copy(self.population[i])
            for j in range(self.dim):
                if np.random.rand() < self.de_crossover_rate:
                    x_trial[j] = x_mutated[j]

            # Selection
            f_trial = func(x_trial)
            self.budget -= 1
            if f_trial < self.fitness[i]:
                self.population[i] = x_trial
                self.fitness[i] = f_trial
                
                # Update SOM with new better solution
                bmu_index = self.find_best_matching_unit(x_trial)
                self.update_som(x_trial, bmu_index)


    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size

        self.initialize_som()
        # Initialize SOM with random vectors from the search space
        for i in range(self.som_grid_size):
          for j in range(self.som_grid_size):
            self.som[i,j] = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.dim))
        
        
        while self.budget > 0:
            self.differential_evolution(func)

            # Update the best solution found so far
            best_index = np.argmin(self.fitness)
            if self.fitness[best_index] < self.f_opt:
                self.f_opt = self.fitness[best_index]
                self.x_opt = self.population[best_index]

            # Adapt learning rate and neighborhood radius (optional)
            self.learning_rate = 0.95 * self.learning_rate
            self.neighborhood_radius = max(1, int(0.95 * self.neighborhood_radius))


        return self.f_opt, self.x_opt