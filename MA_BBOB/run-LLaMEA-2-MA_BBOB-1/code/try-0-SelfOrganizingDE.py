import numpy as np

class SelfOrganizingDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, initial_F=0.5, initial_CR=0.7, diversity_threshold=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = initial_F
        self.CR = initial_CR
        self.diversity_threshold = diversity_threshold
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
        self.lb = func.bounds.lb
        self.ub = func.bounds.ub

    def calculate_diversity(self):
        """Calculates the average distance of each individual from the population mean."""
        mean_position = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - mean_position, axis=1)
        return np.mean(distances)

    def adjust_parameters(self):
        """Adjusts F and CR based on population diversity."""
        diversity = self.calculate_diversity()
        if diversity < self.diversity_threshold:
            # Population is converging, increase exploration
            self.F *= 1.1  # Increase F
            self.CR *= 0.9 # Decrease CR
        else:
            # Population is diverse, increase exploitation
            self.F *= 0.9  # Decrease F
            self.CR *= 1.1 # Increase CR

        self.F = np.clip(self.F, 0.1, 0.9)
        self.CR = np.clip(self.CR, 0.1, 0.9)

    def __call__(self, func):
        self.initialize_population(func)

        while self.budget > 0:
            self.adjust_parameters() # Adjust F and CR dynamically
            for i in range(self.pop_size):
                # Mutation
                indices = np.random.choice(self.pop_size, 3, replace=False)
                x_r1, x_r2, x_r3 = self.population[indices]
                v = self.population[i] + self.F * (x_r2 - x_r3)
                v = np.clip(v, self.lb, self.ub)

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
                        self.best_index = np.argmin(self.fitness)

        return self.f_opt, self.x_opt