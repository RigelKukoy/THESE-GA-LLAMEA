import numpy as np

class SOMAdaptiveDELocalSearch:
    def __init__(self, budget=10000, dim=10, pop_size=50, som_size=10, learning_rate=0.1, local_search_radius=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.som_size = som_size
        self.learning_rate = learning_rate
        self.local_search_radius = local_search_radius
        self.som = np.random.uniform(-5.0, 5.0, size=(self.som_size, self.dim))  # Simplified SOM initialization
        self.F = 0.5
        self.CR = 0.9

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size

        best_idx = np.argmin(fitness)
        if fitness[best_idx] < self.f_opt:
            self.f_opt = fitness[best_idx]
            self.x_opt = self.population[best_idx]

        while self.budget > 0:
            # SOM Training
            for x in self.population:
                closest_node_idx = np.argmin(np.linalg.norm(self.som - x, axis=1))
                self.som[closest_node_idx] += self.learning_rate * (x - self.som[closest_node_idx])

            for i in range(self.pop_size):
                # Parameter Adaptation based on SOM distance
                closest_node_idx = np.argmin(np.linalg.norm(self.som - self.population[i], axis=1))
                distance = np.linalg.norm(self.som[closest_node_idx] - self.population[i])
                self.F = 0.1 + 0.8 * np.exp(-distance)  # F decreases with distance
                self.CR = 0.1 + 0.8 * (1 - np.exp(-distance)) # CR increases with distance


                # Mutation
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                x_1, x_2, x_3 = self.population[idxs]
                mutant = x_1 + self.F * (x_2 - x_3)
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                # Crossover
                crossover = np.random.uniform(size=self.dim) < self.CR
                trial = np.where(crossover, mutant, self.population[i])

                # Selection
                f_trial = func(trial)
                self.budget -= 1

                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial

                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    self.population[i] = trial
            
            # Local Search around best solution
            if self.budget > 0:
                x_local = self.x_opt + np.random.uniform(-self.local_search_radius, self.local_search_radius, size=self.dim)
                x_local = np.clip(x_local, func.bounds.lb, func.bounds.ub)
                f_local = func(x_local)
                self.budget -= 1
                if f_local < self.f_opt:
                    self.f_opt = f_local
                    self.x_opt = x_local

        return self.f_opt, self.x_opt