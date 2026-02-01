import numpy as np

class DynamicDE:
    def __init__(self, budget=10000, dim=10, initial_pop_size=20, min_pop_size=5, max_pop_size=100, adapt_freq=10):
        self.budget = budget
        self.dim = dim
        self.pop_size = initial_pop_size
        self.min_pop_size = min_pop_size
        self.max_pop_size = max_pop_size
        self.adapt_freq = adapt_freq
        self.F = 0.5
        self.CR = 0.7
        self.CMA_mu = int(self.pop_size / 4)
        self.CMA_sigma = 0.1
        self.CMA_C = np.eye(self.dim)
        self.CMA_d = np.ones(self.dim)
        self.CMA_eigenspace = np.eye(self.dim)
        self.t = 0

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        self.best_fitness_history = []

        while self.budget > 0:
            for i in range(self.pop_size):
                if self.budget <= 0:
                    break

                # Mutation using CMA
                z = np.random.normal(0, 1, self.dim)
                mutant = self.population[i] + self.CMA_sigma * self.CMA_eigenspace.dot(self.CMA_d * z)
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                # Crossover
                trial = np.copy(self.population[i])
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.CR or j == j_rand:
                        trial[j] = mutant[j]

                # Selection
                f = func(trial)
                self.budget -= 1
                if f < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = f

                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = trial
            
            self.best_fitness_history.append(self.f_opt)

            # Population size adaptation
            if self.t % self.adapt_freq == 0:
                if len(self.best_fitness_history) > self.adapt_freq:
                    improvement = self.best_fitness_history[-self.adapt_freq-1] - self.best_fitness_history[-1]
                    if improvement > 0:
                        self.pop_size = min(self.pop_size + 1, self.max_pop_size)
                    else:
                        self.pop_size = max(self.pop_size - 1, self.min_pop_size)

                    if self.pop_size != self.population.shape[0]:
                        # Resize population (keep best individuals)
                        best_indices = np.argsort(self.fitness)[:self.pop_size]
                        self.population = self.population[best_indices]
                        self.fitness = self.fitness[best_indices]
                        while self.population.shape[0] < self.pop_size:
                            x = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(1, self.dim))
                            f = func(x[0])
                            self.budget -= 1
                            self.population = np.vstack((self.population, x))
                            self.fitness = np.append(self.fitness, f)


            # CMA Update
            if self.t % self.adapt_freq == 0:
              
                # Sort population based on fitness
                sorted_indices = np.argsort(self.fitness)
                mu_individuals = self.population[sorted_indices[:self.CMA_mu]]
                
                # Calculate the weighted mean of the selected individuals
                weights = np.log(self.CMA_mu + 0.5) - np.log(np.arange(1, self.CMA_mu + 1))
                weights /= np.sum(weights)
                mean = np.sum(mu_individuals * weights[:, np.newaxis], axis=0)
                
                # Update covariance matrix
                C = np.zeros_like(self.CMA_C)
                for k in range(self.CMA_mu):
                    diff = mu_individuals[k] - mean
                    C += weights[k] * np.outer(diff, diff)
                self.CMA_C = C

                # Eigendecomposition
                self.CMA_d, self.CMA_eigenspace = np.linalg.eigh(self.CMA_C)
                self.CMA_d = np.sqrt(np.maximum(self.CMA_d, 0)) # Ensure positive values
                

            self.t += 1

        return self.f_opt, self.x_opt