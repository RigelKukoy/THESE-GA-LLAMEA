import numpy as np

class DynamicPopMirroredDE:
    def __init__(self, budget=10000, dim=10, initial_pop_size=100, CR=0.5, F=0.7, reduction_factor=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = initial_pop_size
        self.initial_pop_size = initial_pop_size
        self.CR = CR
        self.F = F
        self.reduction_factor = reduction_factor
        self.population = None
        self.fitness = None
        self.f_opt = np.Inf
        self.x_opt = None
        self.eval_count = 0
        self.stagnation_counter = 0
        self.stagnation_threshold = 50  # Number of iterations without improvement to trigger reduction

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.eval_count += self.pop_size

        best_index = np.argmin(self.fitness)
        if self.fitness[best_index] < self.f_opt:
            self.f_opt = self.fitness[best_index]
            self.x_opt = self.population[best_index].copy()

    def evolve(self, func):
        new_population = np.zeros_like(self.population)
        new_fitness = np.zeros_like(self.fitness)

        for i in range(self.pop_size):
            # Mutation
            idxs = np.random.choice(self.pop_size, 3, replace=False)
            x_r1, x_r2, x_r3 = self.population[idxs]
            x_mutated = x_r1 + self.F * (x_r2 - x_r3)
            x_mutated = np.clip(x_mutated, func.bounds.lb, func.bounds.ub)

            # Crossover
            x_trial = self.population[i].copy()
            j_rand = np.random.randint(self.dim)
            for j in range(self.dim):
                if np.random.rand() < self.CR or j == j_rand:
                    x_trial[j] = x_mutated[j]

            # Simplified Mirrored Sampling
            x_mirrored = 2 * self.x_opt - x_trial  # Mirror around current best
            x_mirrored = np.clip(x_mirrored, func.bounds.lb, func.bounds.ub)

            # Selection: Compare trial and mirrored vectors
            f_trial = func(x_trial)
            f_mirrored = func(x_mirrored)
            self.eval_count += 2

            if f_trial < f_mirrored:
                new_population[i] = x_trial
                new_fitness[i] = f_trial
            else:
                new_population[i] = x_mirrored
                new_fitness[i] = f_mirrored

            if new_fitness[i] < self.f_opt:
                self.f_opt = new_fitness[i]
                self.x_opt = new_population[i].copy()
                self.stagnation_counter = 0 # Reset stagnation counter
        
        # Update population and fitness
        self.population = new_population
        self.fitness = new_fitness

        # Stagnation check and population reduction
        self.stagnation_counter += 1
        if self.stagnation_counter > self.stagnation_threshold and self.pop_size > 10:  # Minimum pop size of 10
            self.reduce_population()
            self.stagnation_counter = 0

    def reduce_population(self):
        new_pop_size = int(self.pop_size * self.reduction_factor)
        if new_pop_size < 10:
            new_pop_size = 10
            
        # Sort population by fitness
        sorted_indices = np.argsort(self.fitness)
        
        # Keep the best individuals
        self.population = self.population[sorted_indices[:new_pop_size]]
        self.fitness = self.fitness[sorted_indices[:new_pop_size]]
        self.pop_size = new_pop_size


    def __call__(self, func):
        self.initialize_population(func)
        while self.eval_count < self.budget:
            self.evolve(func)
            if self.eval_count >= self.budget:
                break
        return self.f_opt, self.x_opt