import numpy as np

class SOMGuidedSearch:
    def __init__(self, budget=10000, dim=10, som_size=10, learning_rate=0.1, neighborhood_radius=None, local_search_iterations=5):
        self.budget = budget
        self.dim = dim
        self.som_size = som_size
        self.learning_rate = learning_rate
        self.neighborhood_radius = neighborhood_radius if neighborhood_radius is not None else som_size // 3
        self.local_search_iterations = local_search_iterations

        self.som = np.random.uniform(-1, 1, size=(som_size, som_size, dim)) # Initialize SOM weights
        self.fitness_map = np.full((som_size, som_size), np.inf)

        self.x_opt = None
        self.f_opt = np.inf

    def find_best_matching_unit(self, x):
        """Find the best matching unit (BMU) in the SOM for a given input vector."""
        distances = np.sum((self.som - x)**2, axis=2)
        return np.unravel_index(np.argmin(distances), distances.shape)

    def update_som(self, x, bmu, func):
         """Update the SOM weights based on the input vector and BMU."""
         for i in range(self.som_size):
            for j in range(self.som_size):
                distance = np.sqrt((i - bmu[0])**2 + (j - bmu[1])**2)
                if distance <= self.neighborhood_radius:
                    influence = np.exp(-distance**2 / (2 * self.neighborhood_radius**2))
                    self.som[i, j] += self.learning_rate * influence * (x - self.som[i, j])
                    f = func(self.som[i,j])
                    if f < self.fitness_map[i,j]:
                      self.fitness_map[i,j] = f


    def local_search(self, x, func):
        """Perform a local search around a given point."""
        best_x = x.copy()
        best_f = func(x)
        for _ in range(self.local_search_iterations):
            # Generate a random perturbation
            perturbation = np.random.normal(0, 0.1, size=self.dim)
            new_x = x + perturbation
            new_x = np.clip(new_x, func.bounds.lb, func.bounds.ub)
            new_f = func(new_x)

            if new_f < best_f:
                best_f = new_f
                best_x = new_x

        return best_f, best_x


    def __call__(self, func):
        # Initial exploration
        initial_samples = min(self.budget, self.som_size * self.som_size) # limit initial sampling to budget
        for _ in range(initial_samples):
            x = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
            f = func(x)
            self.budget -= 1

            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x

            bmu = self.find_best_matching_unit(x)
            self.update_som(x, bmu, func)

        # Main optimization loop
        while self.budget > 0:
            # Select a random unit from the SOM
            i, j = np.random.randint(0, self.som_size, size=2)
            x = self.som[i, j].copy()

            # Perform local search
            f_local, x_local = self.local_search(x, func)
            self.budget -= self.local_search_iterations # Account for local search evals.

            if f_local < self.f_opt:
                self.f_opt = f_local
                self.x_opt = x_local

            # Update the SOM based on the local search result.
            bmu = self.find_best_matching_unit(x_local)
            self.update_som(x_local, bmu, func)

        return self.f_opt, self.x_opt