import numpy as np
from scipy.optimize import minimize

class SOMNelderMead:
    def __init__(self, budget=10000, dim=10, pop_size=10, som_grid_size=3):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.som_grid_size = som_grid_size
        self.som = None
        self.learning_rate = 0.1
        self.neighborhood_radius = som_grid_size // 2
        self.min_inner_budget = 100  # Minimum function evaluations for Nelder-Mead

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

    def nelder_mead(self, func, x0, inner_budget):
        """Applies the Nelder-Mead optimization algorithm."""
        if inner_budget <= 0:
             return func(x0), x0

        bounds = func.bounds
        
        def constrained_func(x):
            # Clip the solution within the bounds
            x_clipped = np.clip(x, bounds.lb, bounds.ub)
            return func(x_clipped) # Evaluate the original function within the bounds

        result = minimize(constrained_func, x0, method='Nelder-Mead', options={'maxfev': inner_budget, 'adaptive': True})

        return result.fun, result.x

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None

        # Initialize population with random samples
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size

        self.initialize_som()

        # Initialize SOM with random vectors from the search space
        for i in range(self.som_grid_size):
            for j in range(self.som_grid_size):
                self.som[i, j] = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.dim))
        
        while self.budget > self.min_inner_budget:
            # Select individuals from population and SOM for Nelder-Mead
            candidates = []
            
            # Add individuals from population
            selected_indices = np.random.choice(self.pop_size, size=min(self.pop_size, 3), replace=False)
            candidates.extend(population[selected_indices])

            # Add individuals from SOM
            som_indices = np.random.choice(self.som_grid_size * self.som_grid_size, size=min(self.som_grid_size * self.som_grid_size, 3), replace=False)
            som_coords = np.unravel_index(som_indices, (self.som_grid_size, self.som_grid_size))
            for k in range(len(som_indices)):
                candidates.append(self.som[som_coords[0][k], som_coords[1][k]])


            for x0 in candidates:
                inner_budget = min(self.budget, self.min_inner_budget + int(self.budget / (len(candidates) + 1)))
                f_val, x_new = self.nelder_mead(func, x0, inner_budget)
                self.budget -= inner_budget
                
                # Update SOM
                bmu_index = self.find_best_matching_unit(x_new)
                self.update_som(x_new, bmu_index)


                if f_val < self.f_opt:
                    self.f_opt = f_val
                    self.x_opt = x_new

                #Update population if it is better than the worst
                if f_val < np.max(fitness):
                  worst_index = np.argmax(fitness)
                  population[worst_index] = x_new
                  fitness[worst_index] = f_val
        
            # Adapt learning rate and neighborhood radius (optional)
            self.learning_rate = 0.95 * self.learning_rate
            self.neighborhood_radius = max(1, int(0.95 * self.neighborhood_radius))


        return self.f_opt, self.x_opt