import numpy as np

class AdaptiveDERS:
    def __init__(self, budget=10000, dim=10, initial_pop_size=50, min_pop_size=10, max_pop_size=100, F=0.5, CR=0.7, diversity_threshold=0.01, restart_frequency=500):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = initial_pop_size
        self.min_pop_size = min_pop_size
        self.max_pop_size = max_pop_size
        self.pop_size = initial_pop_size
        self.F = F
        self.CR = CR
        self.diversity_threshold = diversity_threshold
        self.restart_frequency = restart_frequency
        self.population = None
        self.fitness = None
        self.best_index = None
        self.f_opt = np.inf
        self.x_opt = None
        self.eval_count = 0
        self.generation = 0

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.eval_count += self.pop_size
        self.best_index = np.argmin(self.fitness)
        self.f_opt = self.fitness[self.best_index]
        self.x_opt = self.population[self.best_index].copy()

    def calculate_diversity(self):
        distances = np.linalg.norm(self.population - np.mean(self.population, axis=0), axis=1)
        diversity = np.std(distances)
        return diversity

    def adjust_population_size(self):
        diversity = self.calculate_diversity()
        if diversity < self.diversity_threshold:
            self.pop_size = max(self.min_pop_size, int(self.pop_size * 0.8))  # Reduce population size
        else:
            self.pop_size = min(self.max_pop_size, int(self.pop_size * 1.2))  # Increase population size
        self.pop_size = int(self.pop_size) # Ensure integer pop_size


    def restart_population(self, func):
         self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
         self.fitness = np.array([func(x) for x in self.population])
         self.eval_count += self.pop_size
         best_index = np.argmin(self.fitness)
         if self.fitness[best_index] < self.f_opt:
            self.f_opt = self.fitness[best_index]
            self.x_opt = self.population[best_index].copy()


    def __call__(self, func):
        self.initialize_population(func)

        while self.eval_count < self.budget:
            self.generation += 1
            for i in range(self.pop_size):
                # Mutation
                indices = np.random.choice(self.pop_size, 3, replace=False)
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
                self.eval_count += 1

                # Selection
                if f_u < self.fitness[i]:
                    self.fitness[i] = f_u
                    self.population[i] = u
                    if f_u < self.f_opt:
                        self.f_opt = f_u
                        self.x_opt = u.copy()

                if self.eval_count >= self.budget:
                    break
            
            if self.generation % 10 == 0:  # Adjust population every 10 generations
                self.adjust_population_size()

            if self.generation % self.restart_frequency == 0:
                self.restart_population(func)

        return self.f_opt, self.x_opt