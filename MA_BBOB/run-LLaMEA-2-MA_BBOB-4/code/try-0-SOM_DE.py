import numpy as np

class SOM_DE:
    def __init__(self, budget=10000, dim=10, pop_size=40, F=0.5, Cr=0.9, som_grid_size=5, som_learning_rate=0.1, som_sigma=1.0):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.Cr = Cr
        self.som_grid_size = som_grid_size
        self.som_learning_rate = som_learning_rate
        self.som_sigma = som_sigma
        self.som_weights = np.random.rand(som_grid_size, som_grid_size, dim)  # SOM weights
        self.best_fitness_history = []
        self.clusters = np.zeros(pop_size, dtype=int) # Cluster assignment for each individual

    def find_closest_node(self, x):
        """Finds the closest SOM node to the input vector x."""
        distances = np.sum((self.som_weights - x)**2, axis=2)
        return np.unravel_index(np.argmin(distances), distances.shape[:2])

    def update_som(self, x, winning_node):
        """Updates the SOM weights based on the winning node and neighborhood."""
        for i in range(self.som_grid_size):
            for j in range(self.som_grid_size):
                distance = np.sqrt((i - winning_node[0])**2 + (j - winning_node[1])**2)
                influence = np.exp(-distance**2 / (2 * self.som_sigma**2))
                self.som_weights[i, j] += self.som_learning_rate * influence * (x - self.som_weights[i, j])
                
    def assign_clusters(self, population):
        """Assigns each individual to its closest SOM node."""
        for i, x in enumerate(population):
            self.clusters[i] = np.ravel_multi_index(self.find_closest_node(x), (self.som_grid_size, self.som_grid_size))

    def __call__(self, func):
        # Initialization
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size
        
        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]
        self.best_fitness_history.append(self.f_opt)
        
        generation = 0

        while self.budget > self.pop_size:
            # Update SOM and assign clusters
            for x in population:
                winning_node = self.find_closest_node(x)
                self.update_som(x, winning_node)
            self.assign_clusters(population)
            
            # Mutation and Crossover
            new_population = np.copy(population)
            for i in range(self.pop_size):
                # Cluster-based mutation strategy (example: different F values)
                cluster = self.clusters[i]
                if cluster % 3 == 0:
                    F = self.F  # Standard F
                elif cluster % 3 == 1:
                    F = self.F * 0.8  # Smaller F
                else:
                    F = self.F * 1.2  # Larger F
                
                indices = [j for j in range(self.pop_size) if j != i]
                idxs = np.random.choice(indices, size=3, replace=False)
                x_r1, x_r2, x_r3 = population[idxs[0]], population[idxs[1]], population[idxs[2]]
                
                mutant = population[i] + F * (x_r1 - x_r2)
                
                # Crossover
                for j in range(self.dim):
                    if np.random.rand() < self.Cr:
                        new_population[i, j] = mutant[j]
                    
                new_population[i] = np.clip(new_population[i], func.bounds.lb, func.bounds.ub)

            # Evaluation
            new_fitness = np.array([func(x) for x in new_population])
            self.budget -= self.pop_size

            # Selection
            for i in range(self.pop_size):
                if new_fitness[i] < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]
                    
                    if new_fitness[i] < self.f_opt:
                        self.f_opt = new_fitness[i]
                        self.x_opt = new_population[i]
                        
            self.best_fitness_history.append(self.f_opt)
            generation += 1

        return self.f_opt, self.x_opt