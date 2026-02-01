import numpy as np

class AdaptiveGaussianSearch:
    def __init__(self, budget=10000, dim=10, pop_size=20, initial_std=1.0, learning_rate=0.1, success_rate_alpha=0.1, bound_shrink_factor=0.999):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.initial_std = initial_std
        self.learning_rate = learning_rate
        self.lb = -5.0
        self.ub = 5.0
        self.success_rate_alpha = success_rate_alpha
        self.bound_shrink_factor = bound_shrink_factor
        self.std = initial_std
        self.success_rate = 0.5
        self.eval_count = 0
        self.mean = np.zeros(dim)
        self.covariance = np.eye(dim) * initial_std**2 # Initialize covariance matrix
        self.eigenvalues = None
        self.eigenvectors = None

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        self.eval_count = 0

        # Orthogonal initialization
        population = self.orthogonal_initialization()
        fitness = np.array([func(x) for x in population])
        self.eval_count += self.pop_size

        best_index = np.argmin(fitness)
        if fitness[best_index] < self.f_opt:
            self.f_opt = fitness[best_index]
            self.x_opt = population[best_index]
            self.mean = self.x_opt.copy()

        while self.eval_count < self.budget:
            # Generate offspring using MVN with covariance matrix
            offspring = np.random.multivariate_normal(self.mean, self.covariance, size=self.pop_size)

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
                self.mean = self.x_opt.copy()

            # Selection and update mean
            num_improved = 0
            for i in range(self.pop_size):
                if offspring_fitness[i] < fitness[i]:
                    population[i] = offspring[i].copy()
                    fitness[i] = offspring_fitness[i]
                    num_improved += 1

            # Update success rate
            self.success_rate = (1 - self.success_rate_alpha) * self.success_rate + self.success_rate_alpha * (num_improved / self.pop_size)

            # Covariance matrix adaptation (CMA) inspired update
            diff = population - self.mean
            weighted_diff = (fitness - np.mean(fitness)) * diff
            delta_mean = self.learning_rate * np.mean(diff, axis=0)
            self.mean += delta_mean

            C = np.cov(population.T)
            self.covariance = (1 - self.learning_rate) * self.covariance + self.learning_rate * C

            # Bound adaptation (shrinking)
            self.lb = self.bound_shrink_factor * self.lb + (1 - self.bound_shrink_factor) * self.x_opt
            self.ub = self.bound_shrink_factor * self.ub + (1 - self.bound_shrink_factor) * self.x_opt

            population = np.clip(population, self.lb, self.ub)

            if self.eval_count >= self.budget:
                break
        return self.f_opt, self.x_opt

    def orthogonal_initialization(self):
        # Generate an orthogonal matrix using QR decomposition
        H = np.random.randn(self.pop_size, self.dim)
        Q, R = np.linalg.qr(H)
        
        # Scale each row randomly within bounds
        population = np.zeros((self.pop_size, self.dim))
        for i in range(self.pop_size):
            random_scale = np.random.uniform(0, 1)
            population[i] = self.lb + (self.ub - self.lb) * random_scale * Q[i] #Using a random scale and then Q
            
            #Alternative: sample normally with covariance
            #population = np.random.multivariate_normal(mean=np.zeros(self.dim), cov=np.eye(self.dim), size=self.pop_size)
            #scale appropriately
            
        return np.clip(population, self.lb, self.ub)