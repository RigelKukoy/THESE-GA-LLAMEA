import numpy as np

class DistanceAdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, F_init=0.5, CR_init=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F_init = F_init  # Initial mutation factor
        self.CR_init = CR_init  # Initial crossover rate
        self.population = None
        self.fitness = None
        self.F = None
        self.CR = None
        self.f_opt = np.Inf
        self.x_opt = None
        self.eval_count = 0

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.eval_count += self.pop_size
        self.F = np.full(self.pop_size, self.F_init)
        self.CR = np.full(self.pop_size, self.CR_init)
        
        best_index = np.argmin(self.fitness)
        if self.fitness[best_index] < self.f_opt:
            self.f_opt = self.fitness[best_index]
            self.x_opt = self.population[best_index].copy()

    def distance_based_mutation_factor(self, x, population):
        distances = np.linalg.norm(population - x, axis=1)
        distances = distances / np.sum(distances)  # Normalize to create a probability distribution
        #Weight F by the inverse distance from current solution to rest of population
        return np.clip(np.sum(distances * self.F), 0.1, 1.0)

    def orthogonal_crossover(self, x_mutated, x_target):
        # Generate orthogonal matrix (e.g., using Hadamard matrix if dim is a power of 2)
        if (self.dim & (self.dim - 1) == 0) and self.dim > 1:  # Check if dim is a power of 2
            H = self.hadamard(self.dim)
            # Select one row of Hadamard matrix for crossover
            idx = np.random.randint(self.dim)
            crossover_pattern = (H[idx] + 1) / 2  # Convert -1/1 to 0/1
        else:
            crossover_pattern = np.random.choice([0, 1], size=self.dim)
        
        x_trial = x_target.copy()
        for j in range(self.dim):
            if crossover_pattern[j] == 1:
                x_trial[j] = x_mutated[j]
        return x_trial

    def hadamard(self, n):
        if n == 1:
            return np.array([[1]])
        H = self.hadamard(n // 2)
        return np.vstack((
            np.hstack((H, H)),
            np.hstack((H, -H))
        ))

    def evolve(self, func):
        for i in range(self.pop_size):
            # Adaptive F based on distance to other solutions
            self.F[i] = self.distance_based_mutation_factor(self.population[i], self.population)
            self.CR[i] = np.clip(np.random.normal(self.CR[i], 0.1), 0.1, 1.0)

            # Mutation
            idxs = np.random.choice(self.pop_size, 3, replace=False)
            x_r1, x_r2, x_r3 = self.population[idxs]
            x_mutated = x_r1 + self.F[i] * (x_r2 - x_r3)
            x_mutated = np.clip(x_mutated, func.bounds.lb, func.bounds.ub)

            # Orthogonal Crossover
            x_trial = self.orthogonal_crossover(x_mutated, self.population[i])
            x_trial = np.clip(x_trial, func.bounds.lb, func.bounds.ub)

            # Selection
            f_trial = func(x_trial)
            self.eval_count += 1

            if f_trial < self.fitness[i]:
                self.population[i] = x_trial
                self.fitness[i] = f_trial
                
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = x_trial.copy()

            if self.eval_count >= self.budget:
                break

    def __call__(self, func):
        self.initialize_population(func)
        while self.eval_count < self.budget:
            self.evolve(func)
        return self.f_opt, self.x_opt