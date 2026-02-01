import numpy as np

class AdaptiveCovarianceGaussianSearchMomentum:
    def __init__(self, budget=10000, dim=10, pop_size=20, lr=0.1, momentum=0.9, cs=0.3, damp=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.lr = lr  # Learning rate for step size adaptation
        self.step_size = 1.0  # Initial step size
        self.momentum = momentum
        self.velocity = np.zeros(dim) # Initialize velocity for momentum
        self.C = np.eye(dim)  # Covariance matrix
        self.ps = np.zeros(dim)  # Evolution path for step size
        self.cs = cs  # Step-size damping parameter
        self.damp = damp
        self.mean = None

    def __call__(self, func):
        # Initialize population within bounds
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size
        self.mean = population[np.argmin(fitness)].copy()

        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)].copy()

        generation = 0
        while self.budget > 0:
            generation += 1
            # Mutation: Gaussian perturbation with adaptive step size and momentum
            z = np.random.normal(0, 1, size=(self.pop_size, self.dim))
            mutation = self.step_size * np.dot(z, np.linalg.cholesky(self.C).T)

            # Update velocity with momentum
            self.velocity = self.momentum * self.velocity + (1 - self.momentum) * np.mean(mutation, axis=0)

            offspring = population + mutation #+ self.velocity # Adding momentum to the offspring

            # Clip offspring to remain within bounds
            offspring = np.clip(offspring, func.bounds.lb, func.bounds.ub)

            # Evaluate offspring
            offspring_fitness = np.array([func(x) for x in offspring])
            self.budget -= self.pop_size
            if self.budget <= 0:
                offspring_fitness = offspring_fitness[:self.pop_size + self.budget]

            # Selection: Replace parents with better offspring
            for i in range(len(offspring_fitness)):
                if offspring_fitness[i] < fitness[i]:
                    fitness[i] = offspring_fitness[i]
                    population[i] = offspring[i]

            # Update best solution
            best_index = np.argmin(fitness)
            if fitness[best_index] < self.f_opt:
                self.f_opt = fitness[best_index]
                self.x_opt = population[best_index].copy()
            
            # Update mean
            old_mean = self.mean.copy()
            self.mean = population[np.argmin(fitness)].copy()

            # Cumulation for step-size control
            y = (self.mean - old_mean) / self.step_size
            self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs)) * y
            
            # Step-size adaptation
            self.step_size *= np.exp((self.cs / self.damp) * (np.linalg.norm(self.ps) / np.sqrt(self.dim) - 1))
            self.step_size = max(self.step_size, 1e-6)
            
            # Covariance matrix adaptation
            delta = (mutation / self.step_size)
            
            d_mean = np.mean(delta, axis=0)

            self.C = (1-self.lr)*self.C + self.lr * np.outer(d_mean, d_mean)
                
            # Make sure C is positive definite
            try:
                np.linalg.cholesky(self.C)
            except np.linalg.LinAlgError:
                self.C = np.eye(self.dim)
                

        return self.f_opt, self.x_opt