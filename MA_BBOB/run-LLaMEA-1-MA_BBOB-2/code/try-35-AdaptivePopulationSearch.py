import numpy as np

class AdaptivePopulationSearch:
    def __init__(self, budget=10000, dim=10, initial_pop_size=20, exploration_rate=0.5, local_search_prob=0.1, stagnation_threshold=100, pop_size_multiplier=1.2):
        self.budget = budget
        self.dim = dim
        self.pop_size = initial_pop_size
        self.initial_pop_size = initial_pop_size
        self.exploration_rate = exploration_rate
        self.lb = -5.0
        self.ub = 5.0
        self.local_search_prob = local_search_prob
        self.stagnation_threshold = stagnation_threshold
        self.stagnation_counter = 0
        self.pop_size_multiplier = pop_size_multiplier
        self.mean = np.zeros(dim)
        self.covariance = np.eye(dim)

        self.population = self._orthogonal_initialization(self.pop_size, dim)


    def _orthogonal_initialization(self, pop_size, dim):
        """Initialize population using orthogonal sampling."""
        if pop_size <= dim:
            # Latin Hypercube Sampling if pop_size <= dim
            population = np.zeros((pop_size, dim))
            for j in range(dim):
                permutation = np.random.permutation(pop_size)
                population[:, j] = (self.lb + (self.ub - self.lb) * (permutation + np.random.rand(pop_size)) / pop_size)
        else:
            # Orthogonal array construction
            num_arrays = (pop_size + dim - 1) // dim
            arrays = []
            for _ in range(num_arrays):
                orthogonal_array = np.random.uniform(self.lb, self.ub, size=(dim, dim))
                q, _ = np.linalg.qr(orthogonal_array)
                arrays.append(q)

            orthogonal_matrix = np.vstack(arrays)
            population = self.lb + (self.ub - self.lb) * (orthogonal_matrix[:pop_size, :] * 0.5 + 0.5)

        return population

    def __call__(self, func):
        population = self.population.copy() # Use initialized population
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size

        best_index = np.argmin(fitness)
        self.f_opt = fitness[best_index]
        self.x_opt = population[best_index]
        self.mean = self.x_opt.copy()
        
        previous_best_fitness = self.f_opt

        while self.budget > 0:
            # Stagnation detection
            if self.f_opt >= previous_best_fitness:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0

            previous_best_fitness = self.f_opt

            # Adaptive adjustment of exploration rate based on stagnation
            if self.stagnation_counter > self.stagnation_threshold:
                self.exploration_rate = min(self.exploration_rate + 0.2, 0.9) # Increase exploration
                self.pop_size = int(min(self.pop_size * self.pop_size_multiplier, self.initial_pop_size * 5)) # Increase pop size
                self.stagnation_counter = 0 # Reset counter
                self.population = self._orthogonal_initialization(self.pop_size, self.dim)
                population = self.population.copy()
            else:
                self.exploration_rate = max(self.exploration_rate - 0.05, 0.1) # Decrease exploration
                self.pop_size = self.initial_pop_size
                
            new_population = np.copy(population)
            for i in range(self.pop_size):
                if np.random.rand() < self.exploration_rate:
                    # Exploration: Sample from a simplified CMA-ES distribution
                    z = np.random.normal(0, 1, size=self.dim)
                    new_population[i] = self.mean + np.dot(self.covariance, z) * 0.1 * (self.ub - self.lb)
                    new_population[i] = np.clip(new_population[i], self.lb, self.ub)
                else:
                    # Exploitation: Move towards the current best
                    direction = self.x_opt - population[i]
                    new_population[i] = population[i] + 0.1 * direction + np.random.normal(0, 0.01, self.dim) * (self.ub - self.lb)
                    new_population[i] = np.clip(new_population[i], self.lb, self.ub)

                # Local Search around the best solution
                if np.random.rand() < self.local_search_prob:
                    step_size = 0.005 * (self.ub - self.lb)
                    new_population[i] = self.x_opt + np.random.uniform(-step_size, step_size, size=self.dim)
                    new_population[i] = np.clip(new_population[i], self.lb, self.ub)


            new_fitness = np.array([func(x) for x in new_population])
            available_budget = min(self.pop_size, self.budget)
            new_fitness = new_fitness[:available_budget]
            new_population = new_population[:available_budget]
            self.budget -= available_budget
            
            # Update population (replace if better)
            for i in range(available_budget):
                if new_fitness[i] < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]
                    
                    if new_fitness[i] < self.f_opt:
                        self.f_opt = new_fitness[i]
                        self.x_opt = new_population[i]
                        self.mean = self.x_opt.copy()
            
            if available_budget > 0:
                self.covariance = np.cov(population[:available_budget].T)
                if np.isnan(self.covariance).any() or np.isinf(self.covariance).any():
                    self.covariance = np.eye(self.dim)
                self.covariance += np.eye(self.dim) * 1e-6  # Regularization


        return self.f_opt, self.x_opt