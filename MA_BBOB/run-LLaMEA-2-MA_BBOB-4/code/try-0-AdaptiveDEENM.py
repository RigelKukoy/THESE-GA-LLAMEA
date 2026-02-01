import numpy as np

class AdaptiveDEENM:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.5, Cr=0.9, elite_count=2, neighborhood_size=3):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F  # Mutation factor
        self.Cr = Cr  # Crossover rate
        self.elite_count = elite_count
        self.neighborhood_size = neighborhood_size

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.eval_count = self.pop_size

        while self.eval_count < self.budget:
            # Sort population by fitness
            sorted_indices = np.argsort(self.fitness)
            self.population = self.population[sorted_indices]
            self.fitness = self.fitness[sorted_indices]
            
            if self.fitness[0] < self.f_opt:
                self.f_opt = self.fitness[0]
                self.x_opt = self.population[0]

            for i in range(self.pop_size):
                # Mutation - Euclidean Neighborhood Mutation
                neighbors_indices = np.random.choice(self.pop_size, self.neighborhood_size, replace=False)
                
                # Calculate the Euclidean distances
                distances = np.linalg.norm(self.population[neighbors_indices] - self.population[i], axis=1)
                
                # Select two different neighbors based on distances
                idx = np.argsort(distances)[:2]
                x_r1 = self.population[neighbors_indices[idx[0]]]
                x_r2 = self.population[neighbors_indices[idx[1]]]

                x_mutated = self.population[i] + self.F * (x_r1 - x_r2)

                # Crossover
                x_trial = np.copy(self.population[i])
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.Cr or j == j_rand:
                        x_trial[j] = x_mutated[j]

                x_trial = np.clip(x_trial, func.bounds.lb, func.bounds.ub)
                
                # Selection
                f_trial = func(x_trial)
                self.eval_count += 1

                if f_trial < self.fitness[i]:
                    self.fitness[i] = f_trial
                    self.population[i] = x_trial
                
                # Elitism: Preserve the best solutions
                sorted_indices = np.argsort(self.fitness)
                self.population = self.population[sorted_indices]
                self.fitness = self.fitness[sorted_indices]

        return self.f_opt, self.x_opt