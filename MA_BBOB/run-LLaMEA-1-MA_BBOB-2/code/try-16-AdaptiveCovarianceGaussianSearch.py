import numpy as np

class AdaptiveCovarianceGaussianSearch:
    def __init__(self, budget=10000, dim=10, initial_pop_size=20, lr=0.1, initial_step_size=1.0, success_history_length=10):
        self.budget = budget
        self.dim = dim
        self.pop_size = initial_pop_size
        self.min_pop_size = 5
        self.max_pop_size = 50
        self.lr = lr  # Learning rate for step size adaptation
        self.step_size = initial_step_size  # Initial step size
        self.covariance = np.eye(dim)  # Initialize covariance matrix
        self.mean = None
        self.success_history = []
        self.success_history_length = success_history_length

    def __call__(self, func):
        # Initialize population within bounds
        if self.mean is None:
            self.mean = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
        population = np.random.multivariate_normal(self.mean, self.step_size * self.covariance, size=self.pop_size)

        population = np.clip(population, func.bounds.lb, func.bounds.ub)  # Clip population

        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size

        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]

        generation = 0
        while self.budget > 0:
            generation += 1
            # Mutation: Gaussian perturbation with adaptive covariance matrix
            mutation = np.random.multivariate_normal(np.zeros(self.dim), self.step_size * self.covariance, size=self.pop_size)
            offspring = population + mutation

            # Clip offspring to remain within bounds
            offspring = np.clip(offspring, func.bounds.lb, func.bounds.ub)

            # Evaluate offspring
            offspring_fitness = np.array([func(x) for x in offspring])
            remaining_evals = min(self.pop_size, self.budget)
            offspring_fitness = offspring_fitness[:remaining_evals]
            self.budget -= remaining_evals
            if remaining_evals < self.pop_size:
                offspring = offspring[:remaining_evals] # truncate offspring to have correct dimensions as offspring_fitness
                

            # Selection: Replace parents with better offspring
            for i in range(len(offspring_fitness)):
                if offspring_fitness[i] < fitness[i]:
                    fitness[i] = offspring_fitness[i]
                    population[i] = offspring[i]

            # Update best solution
            best_index = np.argmin(fitness)
            if fitness[best_index] < self.f_opt:
                self.f_opt = fitness[best_index]
                self.x_opt = population[best_index]

            # Adapt step size and covariance matrix (simplified CMA-ES like update)
            success_count = np.sum(offspring_fitness < fitness[:len(offspring_fitness)])
            success_rate = success_count / len(offspring_fitness)
            self.success_history.append(success_rate)
            if len(self.success_history) > self.success_history_length:
                self.success_history.pop(0)

            avg_success_rate = np.mean(self.success_history) if self.success_history else 0

            if avg_success_rate > 0.3:
                self.step_size *= (1 + self.lr)  # Increase step size if exploration is promising
                self.pop_size = min(self.pop_size + 1, self.max_pop_size)  # Increase population size
            elif avg_success_rate < 0.15:
                self.step_size *= (1 - self.lr)  # Decrease step size if exploration is not fruitful
                self.pop_size = max(self.pop_size - 1, self.min_pop_size)  # Decrease population size
            
            self.step_size = max(self.step_size, 1e-6)  # Ensure step size doesn't become too small
            self.pop_size = int(self.pop_size)


            # Update covariance matrix (simplified rank-one update)
            diff = population[np.argmin(fitness)] - self.mean
            self.covariance = (1 - self.lr) * self.covariance + self.lr * np.outer(diff, diff)
            # Ensure covariance matrix is positive definite
            self.covariance = (self.covariance + self.covariance.T) / 2
            try:
                np.linalg.cholesky(self.covariance)
            except np.linalg.LinAlgError:
                self.covariance += 1e-6 * np.eye(self.dim)  # Add a small diagonal matrix for stability
            
            self.mean = population[np.argmin(fitness)]


        return self.f_opt, self.x_opt