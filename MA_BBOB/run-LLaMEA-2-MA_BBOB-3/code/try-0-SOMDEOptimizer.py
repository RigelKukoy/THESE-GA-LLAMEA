import numpy as np

class SOMDEOptimizer:
    def __init__(self, budget=10000, dim=10, pop_size=20, som_grid_size=5, learning_rate=0.1, de_mutation_factor_base=0.5):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.som_grid_size = som_grid_size
        self.learning_rate = learning_rate
        self.de_mutation_factor_base = de_mutation_factor_base
        self.population = np.random.uniform(-5, 5, size=(pop_size, dim))
        self.fitness = np.zeros(pop_size)
        self.best_positions = self.population.copy()
        self.best_fitness = np.full(pop_size, np.inf)
        self.global_best_position = None
        self.global_best_fitness = np.inf
        self.eval_count = 0

        # Initialize SOM neurons
        self.som_neurons = np.random.uniform(-5, 5, size=(som_grid_size, som_grid_size, dim))

    def __call__(self, func):
        self.eval_count = 0
        while self.eval_count < self.budget:
            # Evaluate fitness
            for i in range(self.pop_size):
                if self.eval_count < self.budget:
                    f = func(self.population[i])
                    self.eval_count += 1
                    self.fitness[i] = f
                    if f < self.best_fitness[i]:
                        self.best_fitness[i] = f
                        self.best_positions[i] = self.population[i].copy()
                        if f < self.global_best_fitness:
                            self.global_best_fitness = f
                            self.global_best_position = self.population[i].copy()

            # Train SOM
            for i in range(self.pop_size):
                # Find the best matching unit (BMU)
                bmu_index = self.find_bmu(self.population[i])
                
                # Update the SOM neurons
                self.update_som_neurons(self.population[i], bmu_index)

            # Update population using DE with SOM-based mutation factor
            for i in range(self.pop_size):
                r1, r2, r3 = np.random.choice(self.pop_size, 3, replace=False)
                
                # Find BMU for the current individual
                bmu_index = self.find_bmu(self.population[i])
                
                # Calculate distance to other neurons in SOM
                distances = self.calculate_neuron_distances(bmu_index)
                
                # Adapt mutation factor based on distances (e.g., inverse proportional)
                de_mutation_factor = self.de_mutation_factor_base / (1 + np.mean(distances))  # Smaller distances -> larger mutation

                de_vector = self.population[r1] + de_mutation_factor * (self.population[r2] - self.population[r3])
                de_vector = np.clip(de_vector, func.bounds.lb, func.bounds.ub)
                
                # Update individual only if it improves fitness
                f_new = func(de_vector) if self.eval_count < self.budget else np.inf
                if self.eval_count < self.budget:
                    self.eval_count += 1
                    if f_new < self.fitness[i]:
                        self.population[i] = de_vector
                        self.fitness[i] = f_new
                        if f_new < self.best_fitness[i]:
                            self.best_fitness[i] = f_new
                            self.best_positions[i] = self.population[i].copy()
                            if f_new < self.global_best_fitness:
                                self.global_best_fitness = f_new
                                self.global_best_position = self.population[i].copy()

        return self.global_best_fitness, self.global_best_position
    
    def find_bmu(self, individual):
        """Find the best matching unit (BMU) in the SOM grid."""
        min_dist = np.inf
        bmu_index = None
        for i in range(self.som_grid_size):
            for j in range(self.som_grid_size):
                dist = np.linalg.norm(individual - self.som_neurons[i, j])
                if dist < min_dist:
                    min_dist = dist
                    bmu_index = (i, j)
        return bmu_index

    def update_som_neurons(self, individual, bmu_index):
        """Update the SOM neurons based on the BMU and learning rate."""
        for i in range(self.som_grid_size):
            for j in range(self.som_grid_size):
                distance = np.sqrt((i - bmu_index[0])**2 + (j - bmu_index[1])**2)
                influence = np.exp(-distance**2 / (2 * (self.som_grid_size/2)**2))  # Gaussian neighborhood
                self.som_neurons[i, j] += self.learning_rate * influence * (individual - self.som_neurons[i, j])
    
    def calculate_neuron_distances(self, bmu_index):
        """Calculate distances from BMU to all other neurons."""
        distances = []
        for i in range(self.som_grid_size):
            for j in range(self.som_grid_size):
                distances.append(np.sqrt((i - bmu_index[0])**2 + (j - bmu_index[1])**2))
        return np.array(distances)