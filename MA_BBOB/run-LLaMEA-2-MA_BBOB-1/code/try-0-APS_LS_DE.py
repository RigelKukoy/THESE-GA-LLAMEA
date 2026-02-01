import numpy as np

class APS_LS_DE:
    def __init__(self, budget=10000, dim=10, initial_population_size=50, local_search_iterations=5):
        self.budget = budget
        self.dim = dim
        self.population_size = initial_population_size
        self.local_search_iterations = local_search_iterations
        self.F = 0.5  # Differential evolution parameter
        self.CR = 0.7 # Crossover rate

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.population_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.population_size
        self.best_index = np.argmin(self.fitness)
        self.best_x = self.population[self.best_index].copy()
        self.best_f = self.fitness[self.best_index]

    def evolve(self, func):
        for i in range(self.population_size):
            # Mutation
            idxs = np.random.choice(self.population_size, 3, replace=False)
            x_r1, x_r2, x_r3 = self.population[idxs]
            x_mutated = x_r1 + self.F * (x_r2 - x_r3)
            x_mutated = np.clip(x_mutated, func.bounds.lb, func.bounds.ub)

            # Crossover
            x_trial = self.population[i].copy()
            j_rand = np.random.randint(self.dim)
            for j in range(self.dim):
                if np.random.rand() < self.CR or j == j_rand:
                    x_trial[j] = x_mutated[j]

            # Selection
            f_trial = func(x_trial)
            self.budget -= 1
            if f_trial < self.fitness[i]:
                self.fitness[i] = f_trial
                self.population[i] = x_trial
                if f_trial < self.best_f:
                    self.best_f = f_trial
                    self.best_x = x_trial.copy()
                    self.best_index = i

    def adjust_population_size(self):
        # Dynamically adjust population size
        if self.budget > 0:
            if np.std(self.fitness) < 1e-4: # Stagnation
                self.population_size = min(self.population_size * 2, 200) #increase, max pop size of 200
            else:
                self.population_size = max(int(self.population_size * 0.9), 10) #decrease, min pop size of 10
            
            # Resize the population: add new individuals or remove the worst ones
            if self.population_size > self.population.shape[0]:
                 num_new = self.population_size - self.population.shape[0]
                 new_individuals = np.random.uniform(self.population[0], self.population[-1], size=(num_new, self.dim))
                 self.population = np.vstack([self.population, new_individuals])
                 new_fitness = np.array([func(x) for x in new_individuals])
                 self.fitness = np.concatenate([self.fitness, new_fitness])
                 self.budget -= num_new
            elif self.population_size < self.population.shape[0]:
                num_remove = self.population.shape[0] - self.population_size
                worst_indices = np.argsort(self.fitness)[-num_remove:]
                self.population = np.delete(self.population, worst_indices, axis=0)
                self.fitness = np.delete(self.fitness, worst_indices)
            
        if self.budget <= 0:
            self.population_size = 1

    def global_search_boost(self, func):
        # Add a small percentage of random solutions to increase global search
        num_new_solutions = int(0.1 * self.population_size)
        if num_new_solutions > 0 and self.budget > 0:
            new_solutions = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(num_new_solutions, self.dim))
            new_fitness = np.array([func(x) for x in new_solutions])
            self.budget -= num_new_solutions
            self.population = np.vstack([self.population, new_solutions])
            self.fitness = np.concatenate([self.fitness, new_fitness])
            
            # Update best solution
            best_index_new = np.argmin(new_fitness)
            if new_fitness[best_index_new] < self.best_f:
                 self.best_f = new_fitness[best_index_new]
                 self.best_x = new_solutions[best_index_new].copy()

    def local_search(self, func):
        # Perform local search around the best solution
        for _ in range(min(self.local_search_iterations, self.budget)):
            perturbation = np.random.normal(0, 0.01, size=self.dim)
            x_local = self.best_x + perturbation
            x_local = np.clip(x_local, func.bounds.lb, func.bounds.ub)
            f_local = func(x_local)
            self.budget -= 1
            if f_local < self.best_f:
                self.best_f = f_local
                self.best_x = x_local.copy()
                
    def __call__(self, func):
        self.initialize_population(func)

        while self.budget > self.local_search_iterations:
            self.evolve(func)
            self.adjust_population_size()
            self.global_search_boost(func)

        self.local_search(func)

        return self.best_f, self.best_x