import numpy as np

class NeighborhoodDiversityDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.5, CR=0.7, neighborhood_size=5):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.neighborhood_size = neighborhood_size
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

    def calculate_diversity(self):
        """Calculates the average distance between population members."""
        distances = []
        for i in range(self.pop_size):
            for j in range(i + 1, self.pop_size):
                distances.append(np.linalg.norm(self.population[i] - self.population[j]))
        return np.mean(distances) if distances else 0

    def __call__(self, func):
        self.initialize_population(func)

        while self.budget > 0:
            diversity = self.calculate_diversity()
            # Adapt F and CR based on diversity
            self.F = 0.1 + 0.9 * (diversity / (func.bounds.ub[0] - func.bounds.lb[0]))  # Scale F based on diversity
            self.CR = 0.1 + 0.9 * np.exp(-diversity / (func.bounds.ub[0] - func.bounds.lb[0])) # Scale CR based on diversity

            for i in range(self.pop_size):
                # Neighborhood-based mutation
                neighbors_indices = np.random.choice(self.pop_size, self.neighborhood_size, replace=False)
                neighbors = self.population[neighbors_indices]
                
                # Find best neighbor
                best_neighbor_index = np.argmin(self.fitness[neighbors_indices])
                x_best_neighbor = neighbors[best_neighbor_index]

                # Mutation using best neighbor and two other random individuals
                indices = np.random.choice(self.pop_size, 2, replace=False)
                x_r1, x_r2 = self.population[indices]
                v = self.population[i] + self.F * (x_best_neighbor - self.population[i]) + self.F * (x_r1 - x_r2)
                v = np.clip(v, func.bounds.lb, func.bounds.ub)

                # Crossover
                j_rand = np.random.randint(self.dim)
                u = self.population[i].copy()
                for j in range(self.dim):
                    if np.random.rand() < self.CR or j == j_rand:
                        u[j] = v[j]

                # Evaluation
                f_u = func(u)
                self.budget -= 1

                # Selection
                if f_u < self.fitness[i]:
                    self.fitness[i] = f_u
                    self.population[i] = u
                    if f_u < self.f_opt:
                        self.f_opt = f_u
                        self.x_opt = u.copy()

        return self.f_opt, self.x_opt