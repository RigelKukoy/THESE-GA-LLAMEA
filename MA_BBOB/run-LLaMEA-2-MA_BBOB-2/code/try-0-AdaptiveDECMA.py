import numpy as np

class AdaptiveDECMA:
    def __init__(self, budget=10000, dim=10, initial_popsize=None, F=0.5, CR=0.7, learning_rate_sigma=0.1, learning_rate_C=0.1):
        self.budget = budget
        self.dim = dim
        self.initial_popsize = initial_popsize if initial_popsize is not None else 10 * self.dim
        self.popsize = self.initial_popsize
        self.F = F
        self.CR = CR
        self.learning_rate_sigma = learning_rate_sigma
        self.learning_rate_C = learning_rate_C
        self.mean = None
        self.C = None
        self.sigma = 1.0
        self.best_fitness_history = []

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        
        # Initialize mean and covariance matrix
        self.mean = np.random.uniform(lb, ub, size=self.dim)
        self.C = np.eye(self.dim)
        
        self.population = np.random.multivariate_normal(self.mean, self.sigma**2 * self.C, size=self.popsize)
        self.population = np.clip(self.population, lb, ub)
        self.fitness = np.array([func(x) for x in self.population])
        self.eval_count = self.popsize
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]
        self.best_fitness_history.append(self.f_opt)

        while self.eval_count < self.budget:
            # Generate offspring
            offspring = np.random.multivariate_normal(self.mean, self.sigma**2 * self.C, size=self.popsize)
            offspring = np.clip(offspring, lb, ub)

            # Evaluate offspring
            fitness_offspring = np.array([func(x) for x in offspring])
            self.eval_count += self.popsize

            # Selection
            for i in range(self.popsize):
                if fitness_offspring[i] < self.fitness[i]:
                    self.population[i] = offspring[i]
                    self.fitness[i] = fitness_offspring[i]

                    if self.fitness[i] < self.f_opt:
                        self.f_opt = self.fitness[i]
                        self.x_opt = self.population[i]

            # Update mean
            weights = (self.fitness.mean() - self.fitness) / np.std(self.fitness)
            weights = np.maximum(0, weights)  # Truncate negative weights
            weights /= weights.sum()
            
            delta_mean = np.sum((self.population - self.mean).T * weights, axis=1)
            self.mean += self.learning_rate_sigma * delta_mean

            # Update covariance matrix
            z = (self.population - self.mean) / self.sigma
            self.C = (1 - self.learning_rate_C) * self.C + self.learning_rate_C * np.cov(z.T)
            
            # Ensure C is positive semi-definite
            try:
                np.linalg.cholesky(self.C)
            except np.linalg.LinAlgError:
                self.C = np.eye(self.dim)

            # Update step size
            self.sigma *= np.exp(self.learning_rate_sigma * (np.mean(weights) - 1/self.popsize))

            self.best_fitness_history.append(self.f_opt)
            if self.eval_count > self.budget:
                break
                
        return self.f_opt, self.x_opt