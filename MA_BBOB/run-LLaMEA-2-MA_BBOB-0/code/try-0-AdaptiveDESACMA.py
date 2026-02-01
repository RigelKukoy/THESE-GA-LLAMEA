import numpy as np

class AdaptiveDESACMA:
    def __init__(self, budget=10000, dim=10, pop_multiplier=5, initial_F=0.5, initial_CR=0.9, reduction_factor=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = dim * pop_multiplier
        self.F = initial_F  # Initial mutation factor
        self.CR = initial_CR # Crossover rate
        self.mean = np.zeros(dim) # Mean of the population
        self.C = np.eye(dim) # Covariance matrix
        self.c_learn = 0.1 # Learning rate for covariance matrix
        self.mu = self.pop_size // 4 # Number of individuals for updating CMA
        self.restart_trigger = 100 # Number of iterations without improvement before restart
        self.no_improvement_counter = 0
        self.best_fitness_ever = np.inf
        self.lb = -5.0
        self.ub = 5.0
        self.x_opt = None
        self.f_opt = np.inf
        self.reduction_factor = reduction_factor # Factor for population reduction
        self.min_pop_size = 10 # Minimum population size


    def __call__(self, func):
        self.population = np.random.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size

        while self.budget > 0:
            # Sort population by fitness
            sorted_indices = np.argsort(self.fitness)
            self.population = self.population[sorted_indices]
            self.fitness = self.fitness[sorted_indices]

            if self.fitness[0] < self.best_fitness_ever:
                self.best_fitness_ever = self.fitness[0]
                self.x_opt = self.population[0]
                self.f_opt = self.fitness[0]
                self.no_improvement_counter = 0
            else:
                self.no_improvement_counter += 1

            if self.no_improvement_counter > self.restart_trigger:
                # Restart strategy
                self.mean = np.zeros(self.dim)
                self.C = np.eye(self.dim)
                self.population = np.random.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
                self.fitness = np.array([func(x) for x in self.population])
                self.budget -= self.pop_size
                self.no_improvement_counter = 0
                continue

            # Adaptive F and CR
            adaptive_F = np.random.normal(self.F, 0.1)
            adaptive_CR = np.random.normal(self.CR, 0.1)
            adaptive_F = np.clip(adaptive_F, 0.1, 1.0)
            adaptive_CR = np.clip(adaptive_CR, 0.1, 1.0)

            for i in range(self.pop_size):
                # Mutation using SACMA
                z = np.random.multivariate_normal(np.zeros(self.dim), self.C)
                mutant = self.population[i] + adaptive_F * z
                mutant = np.clip(mutant, self.lb, self.ub)
                
                # Weighted Recombination
                weights = np.random.rand(self.dim)
                weights /= np.sum(weights)
                recombined = np.zeros(self.dim)
                for j in range(self.dim):
                    recombined[j] = weights[j] * mutant[j] + (1 - weights[j]) * self.population[i][j]


                # Crossover
                crossover = np.random.rand(self.dim) < adaptive_CR
                trial = np.where(crossover, recombined, self.population[i])

                # Selection
                f_trial = func(trial)
                self.budget -= 1
                if f_trial < self.fitness[i]:
                    self.fitness[i] = f_trial
                    self.population[i] = trial

                if self.budget <= 0:
                    break

            # Update CMA
            self.mean = np.mean(self.population[:self.mu], axis=0)
            diff = self.population[:self.mu] - self.mean
            self.C = (1 - self.c_learn) * self.C + self.c_learn * (1/self.mu) * np.sum([np.outer(diff[i], diff[i]) for i in range(self.mu)], axis=0)
            # Ensure C is positive definite
            try:
                np.linalg.cholesky(self.C)
            except np.linalg.LinAlgError:
                self.C = np.eye(self.dim)

            # Population reduction
            if self.pop_size > self.min_pop_size:
                new_pop_size = int(self.pop_size * self.reduction_factor)
                new_pop_size = max(new_pop_size, self.min_pop_size)
                if new_pop_size < self.pop_size:
                    self.pop_size = new_pop_size
                    sorted_indices = np.argsort(self.fitness)
                    self.population = self.population[sorted_indices[:self.pop_size]]
                    self.fitness = self.fitness[sorted_indices[:self.pop_size]]

        return self.f_opt, self.x_opt