import numpy as np

class AdaptiveGaussianSearchCMA:
    def __init__(self, budget=10000, dim=10, pop_size=20, initial_std=1.0, momentum=0.1, success_rate_alpha=0.1, cma_learning_rate = 0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.initial_std = initial_std
        self.lb = -5.0
        self.ub = 5.0
        self.momentum = momentum
        self.success_rate_alpha = success_rate_alpha
        self.std = initial_std  # Initial step size
        self.velocity = np.zeros((pop_size, dim)) #Initialize velocity
        self.success_rate = 0.5  # Initialize success rate for step size adaptation
        self.cma_learning_rate = cma_learning_rate
        self.C = np.eye(dim) #Covariance Matrix
        self.eval_count = 0


    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        
        # Initialize population
        population = np.random.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.eval_count += self.pop_size
        
        # Find best initial solution
        best_index = np.argmin(fitness)
        if fitness[best_index] < self.f_opt:
            self.f_opt = fitness[best_index]
            self.x_opt = population[best_index]
        
        
        while self.eval_count < self.budget:
            # Generate offspring using Gaussian mutation with momentum and CMA
            z = np.random.normal(0, 1, size=(self.pop_size, self.dim))
            offspring = population + self.std * (z @ np.linalg.cholesky(self.C).T)

            # Clip offspring to stay within bounds
            offspring = np.clip(offspring, self.lb, self.ub)
            
            # Evaluate offspring
            offspring_fitness = np.array([func(x) for x in offspring])
            self.eval_count += self.pop_size
            
            # Update best solution
            best_offspring_index = np.argmin(offspring_fitness)
            if offspring_fitness[best_offspring_index] < self.f_opt:
                self.f_opt = offspring_fitness[best_offspring_index]
                self.x_opt = offspring[best_offspring_index]
            
            # Select survivors (replace the worst individuals with the best offspring)
            num_improved = 0
            for i in range(self.pop_size):
                if offspring_fitness[i] < fitness[i]:
                    population[i] = offspring[i].copy()
                    fitness[i] = offspring_fitness[i]
                    num_improved += 1

            # Update success rate
            self.success_rate = (1 - self.success_rate_alpha) * self.success_rate + self.success_rate_alpha * (num_improved / self.pop_size)

            # Adjust step size based on success rate (CMA-like adaptation)
            if self.success_rate > 0.2:
                self.std *= np.exp(self.cma_learning_rate * self.success_rate)  # Increase step size
            elif self.success_rate < 0.1:
                self.std *= np.exp(-self.cma_learning_rate * (1-self.success_rate))  # Decrease step size

            self.std = min(self.std, self.initial_std) #Cap to avoid excessive std

            #Update Covariance Matrix (simplified CMA-ES update)
            weights = np.zeros(self.pop_size)
            ranked_indices = np.argsort(offspring_fitness)
            mu = self.pop_size // 4 #Selection pressure
            weights[:mu] = np.log(mu+1) - np.log(np.arange(1, mu+1))
            weights = weights / np.sum(weights)

            offspring_centered = offspring - np.mean(offspring, axis=0)
            delta = offspring_centered[ranked_indices[:mu]]
            self.C = (1 - self.cma_learning_rate) * self.C + self.cma_learning_rate * (delta.T @ np.diag(weights[:mu]) @ delta) / (self.std**2)
            
            # Adjust bounds (shrink towards the best solution)
            self.lb = np.maximum(self.lb, self.x_opt - 2.5 * self.std)
            self.ub = np.minimum(self.ub, self.x_opt + 2.5 * self.std)
            
            population = np.clip(population, self.lb, self.ub)
            
            if self.eval_count >= self.budget:
                break
                
        return self.f_opt, self.x_opt