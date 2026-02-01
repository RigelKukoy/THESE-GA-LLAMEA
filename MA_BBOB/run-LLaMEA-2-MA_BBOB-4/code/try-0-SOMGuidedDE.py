import numpy as np

class SOMGuidedDE:
    def __init__(self, budget=10000, dim=10, pop_size=40, Cr=0.9, F=0.5, som_grid_size=10, learning_rate=0.1, sigma_initial=1.0, sigma_decay=0.995):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.Cr = Cr
        self.F = F
        self.som_grid_size = som_grid_size
        self.learning_rate = learning_rate
        self.sigma = sigma_initial
        self.sigma_decay = sigma_decay
        self.som = np.random.uniform(0, 1, size=(som_grid_size, som_grid_size, dim))  # SOM nodes initialized randomly
        self.f_opt = np.inf
        self.x_opt = None

    def find_best_matching_unit(self, vector):
        """Find the best matching unit (BMU) in the SOM grid."""
        distances = np.sum((self.som - vector)**2, axis=2)
        bmu_index = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu_index

    def update_som(self, vector, bmu_index):
        """Update the SOM based on the input vector and BMU."""
        for i in range(self.som_grid_size):
            for j in range(self.som_grid_size):
                distance = np.sqrt((i - bmu_index[0])**2 + (j - bmu_index[1])**2)
                influence = np.exp(-distance**2 / (2 * self.sigma**2))
                self.som[i, j] += self.learning_rate * influence * (vector - self.som[i, j])

    def __call__(self, func):
        # Initialization
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size

        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)].copy()
        
        while self.budget > self.pop_size:
            new_population = np.copy(population)
            
            for i in range(self.pop_size):
                # Mutation guided by SOM
                bmu_index = self.find_best_matching_unit(population[i])
                bmu = self.som[bmu_index[0], bmu_index[1]]
                
                # Select three random indices, excluding the current index 'i'
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                
                mutant = population[idxs[0]] + self.F * (population[idxs[1]] - population[idxs[2]]) + 0.1 * (bmu - population[i])
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)
                
                # Crossover
                for j in range(self.dim):
                    if np.random.rand() < self.Cr:
                        new_population[i, j] = mutant[j]
                    
            # Evaluation
            new_fitness = np.array([func(x) for x in new_population])
            self.budget -= self.pop_size
            
            # Selection and SOM update
            for i in range(self.pop_size):
                if new_fitness[i] < fitness[i]:
                    fitness_diff = fitness[i] - new_fitness[i]

                    bmu_index = self.find_best_matching_unit(population[i])
                    self.update_som(population[i], bmu_index)  # Update SOM with old position

                    population[i] = new_population[i].copy()
                    fitness[i] = new_fitness[i]

                    bmu_index = self.find_best_matching_unit(population[i])
                    self.update_som(population[i], bmu_index)  # Update SOM with new position
                    
                    if new_fitness[i] < self.f_opt:
                        self.f_opt = new_fitness[i]
                        self.x_opt = new_population[i].copy()

            # Decay SOM parameters
            self.sigma *= self.sigma_decay
            self.learning_rate *= 0.99

        return self.f_opt, self.x_opt