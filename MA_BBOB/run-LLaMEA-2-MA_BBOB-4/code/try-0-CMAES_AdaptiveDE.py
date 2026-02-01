import numpy as np

class CMAES_AdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=None, initial_sigma=0.1, restart_trigger=1e-9):
        self.budget = budget
        self.dim = dim
        self.pop_size = 4 * dim if pop_size is None else pop_size  # Adaptive pop size
        self.initial_sigma = initial_sigma
        self.sigma = initial_sigma
        self.mean = None
        self.C = None
        self.restart_trigger = restart_trigger
        self.best_fitness_history = []
        self.exploration_rate = 1.0

    def initialize(self, func):
        self.mean = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
        self.C = np.eye(self.dim)
        self.f_opt = np.inf
        self.x_opt = None

    def sample_population(self, func):
        z = np.random.multivariate_normal(np.zeros(self.dim), self.C, size=self.pop_size)
        population = self.mean + self.sigma * z
        population = np.clip(population, func.bounds.lb, func.bounds.ub)
        return population

    def update_distribution(self, population, fitness):
        # Weighted recombination
        weights = np.maximum(0, np.log(self.pop_size / 2 + 1) - np.log(np.arange(1, self.pop_size + 1)))
        weights /= np.sum(weights)

        sorted_indices = np.argsort(fitness)
        best_individuals = population[sorted_indices]

        new_mean = np.sum(weights[:, np.newaxis] * best_individuals, axis=0)

        # Rank-one update of covariance matrix
        y = best_individuals[0] - self.mean
        self.C = (1 - 0.1) * self.C + 0.1 * np.outer(y / self.sigma, y / self.sigma)

        self.mean = new_mean

    def __call__(self, func):
        self.initialize(func)
        
        evals = 0
        while self.budget - evals > self.pop_size:
            # Sample population
            population = self.sample_population(func)
            
            # Evaluation
            fitness = np.array([func(x) for x in population])
            evals += self.pop_size

            # Update best solution
            best_index = np.argmin(fitness)
            if fitness[best_index] < self.f_opt:
                self.f_opt = fitness[best_index]
                self.x_opt = population[best_index]
            self.best_fitness_history.append(self.f_opt)

            # Update distribution parameters
            self.update_distribution(population, fitness)

            # Adapt step size (sigma)
            self.sigma *= np.exp(0.5 * (np.mean(fitness) - self.f_opt) / np.std(fitness))

            # Adjust exploration rate
            self.exploration_rate *= 0.995
            self.sigma *= self.exploration_rate
            
            # Restart mechanism
            if self.f_opt < self.restart_trigger:
                self.initialize(func)
                self.sigma = self.initial_sigma # reset sigma after restart
                

        return self.f_opt, self.x_opt