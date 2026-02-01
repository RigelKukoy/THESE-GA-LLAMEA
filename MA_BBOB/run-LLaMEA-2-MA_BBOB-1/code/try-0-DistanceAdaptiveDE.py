import numpy as np

class DistanceAdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.5, CR=0.7, pop_size_adaptation_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.pop_size_adaptation_rate = pop_size_adaptation_rate
        self.population = None
        self.fitness = None
        self.best_index = None
        self.f_opt = np.inf
        self.x_opt = None
        self.archive = []

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        self.best_index = np.argmin(self.fitness)
        self.f_opt = self.fitness[self.best_index]
        self.x_opt = self.population[self.best_index].copy()

    def calculate_distance(self, x):
        """Calculates the average Euclidean distance to other population members."""
        distances = np.linalg.norm(self.population - x, axis=1)
        distances = distances[distances > 0] # Avoid distance to self
        return np.mean(distances) if len(distances) > 0 else 0

    def adapt_population_size(self):
        """Dynamically adjust population size based on the diversity of the population."""
        avg_distance = np.mean([self.calculate_distance(x) for x in self.population])

        if avg_distance > 0.1:  # High diversity: Increase population size
            self.pop_size = min(int(self.pop_size * (1 + self.pop_size_adaptation_rate)), 100)
        else:  # Low diversity: Decrease population size
            self.pop_size = max(int(self.pop_size * (1 - self.pop_size_adaptation_rate)), 10)


    def __call__(self, func):
        self.initialize_population(func)

        while self.budget > 0:
            self.adapt_population_size()
            new_population = np.zeros((self.pop_size, self.dim))
            new_fitness = np.zeros(self.pop_size)

            for i in range(self.pop_size):
                # Distance-based mutation
                distances = np.array([self.calculate_distance(x) for x in self.population])
                probabilities = distances / np.sum(distances)
                indices = np.random.choice(self.pop_size, 3, replace=False, p=probabilities)
                x_r1, x_r2, x_r3 = self.population[indices]
                
                v = self.population[i] + self.F * (x_r1 - x_r2)
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
                    new_fitness[i] = f_u
                    new_population[i] = u
                    if f_u < self.f_opt:
                        self.f_opt = f_u
                        self.x_opt = u.copy()
                else:
                    new_fitness[i] = self.fitness[i]
                    new_population[i] = self.population[i]
            
            self.fitness = new_fitness
            self.population = new_population
            self.best_index = np.argmin(self.fitness)
        
        return self.f_opt, self.x_opt