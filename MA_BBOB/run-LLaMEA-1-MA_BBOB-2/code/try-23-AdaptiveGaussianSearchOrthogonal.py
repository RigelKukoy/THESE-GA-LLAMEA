import numpy as np

class AdaptiveGaussianSearchOrthogonal:
    def __init__(self, budget=10000, dim=10, pop_size=20, lr=0.1, orthogonal_sampling=True):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.lr = lr  # Learning rate for step size adaptation
        self.step_size = 1.0  # Initial step size
        self.orthogonal_sampling = orthogonal_sampling
        self.mean = np.zeros(dim) # Initialize mean
        self.covariance = np.eye(dim) # Initialize covariance matrix
        self.min_step_size = 1e-6

    def __call__(self, func):
        # Initialize population within bounds
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size

        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]

        generation = 0
        while self.budget > 0:
            generation += 1

            # Generate offspring using orthogonal sampling or Gaussian sampling
            if self.orthogonal_sampling:
                offspring = self.generate_orthogonal_offspring(population)
            else:
                offspring = population + np.random.normal(0, self.step_size, size=(self.pop_size, self.dim))
            
            # Clip offspring to remain within bounds
            offspring = np.clip(offspring, func.bounds.lb, func.bounds.ub)

            # Evaluate offspring
            offspring_fitness = np.array([func(x) for x in offspring])
            self.budget -= self.pop_size
            if self.budget <= 0:
                offspring_fitness = offspring_fitness[:self.pop_size + self.budget]
                offspring = offspring[:self.pop_size + self.budget]

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
                self.mean = self.x_opt.copy()

            # Adapt step size
            if generation % 10 == 0:  # Adjust step size every 10 generations
                success_rate = np.sum(offspring_fitness < fitness) / self.pop_size
                if success_rate > 0.2:
                    self.step_size *= (1 + self.lr)  # Increase step size if exploration is promising
                    self.lr *= 0.95 # Reduce learning rate when step size increases
                else:
                    self.step_size *= (1 - self.lr)  # Decrease step size if exploration is not fruitful
                    self.lr *= 1.05 # Increase learning rate when step size decreases
                    self.lr = min(self.lr, 0.5) # Cap the learning rate
                self.step_size = max(self.step_size, self.min_step_size)  # Ensure step size doesn't become too small
            
            # Adapt Covariance matrix (simple adaptation)
            diff = population - self.mean
            self.covariance = np.cov(diff.T)
            if np.linalg.det(self.covariance) <= 0:
                self.covariance = np.eye(self.dim) # Reset covariance if it becomes singular

        return self.f_opt, self.x_opt

    def generate_orthogonal_offspring(self, population):
        # Generate orthogonal matrix
        H = np.random.randn(self.pop_size, self.pop_size)
        Q, _ = np.linalg.qr(H)

        # Generate offspring based on orthogonal matrix
        offspring = np.zeros((self.pop_size, self.dim))
        for i in range(self.pop_size):
            mutation = np.random.normal(0, self.step_size, self.dim)
            offspring[i] = population[i] + np.dot(Q[i, :self.dim], mutation[:self.pop_size])
        return offspring