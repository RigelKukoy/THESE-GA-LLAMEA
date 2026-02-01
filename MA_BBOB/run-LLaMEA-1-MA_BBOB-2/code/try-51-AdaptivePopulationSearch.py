import numpy as np

class AdaptivePopulationSearch:
    def __init__(self, budget=10000, dim=10, pop_size=20, initial_exploration_rate=0.5, local_search_prob=0.1, stagnation_threshold=100, constriction_factor=0.729, success_rate_learning_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.exploration_rate = initial_exploration_rate
        self.lb = -5.0
        self.ub = 5.0
        self.local_search_prob = local_search_prob
        self.stagnation_threshold = stagnation_threshold
        self.constriction_factor = constriction_factor
        self.velocities = np.zeros((pop_size, dim))
        self.stagnation_counter = 0
        self.local_search_decay = 0.99  # Decay factor for local search probability
        self.min_local_search_prob = 0.01 # Minimum local search probability
        self.success_rate_learning_rate = success_rate_learning_rate
        self.success_rate = 0.5 # Initial success rate
        self.success_counter = 0

        # Orthogonal initialization for better diversity
        self.population = self._orthogonal_initialization(pop_size, dim)

        # Covariance matrix adaptation
        self.C = np.eye(dim)  # Initialize covariance matrix
        self.learning_rate_C = 0.1  # Learning rate for covariance matrix adaptation
        self.eigenvalues = np.ones(dim)

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
        
        previous_best_fitness = self.f_opt
        self.success_counter = 0 # Reset success counter

        while self.budget > 0:
            # Stagnation detection
            if self.f_opt >= previous_best_fitness:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0

            previous_best_fitness = self.f_opt

            # Adaptive adjustment of exploration rate based on success rate
            if self.success_rate > 0.3:
              self.exploration_rate = max(self.exploration_rate - self.success_rate_learning_rate, 0.1)
            else:
              self.exploration_rate = min(self.exploration_rate + self.success_rate_learning_rate, 0.9)

            new_population = np.copy(population)
            new_fitness = np.zeros(self.pop_size) # Allocate before the loop

            for i in range(self.pop_size):
                if np.random.rand() < self.exploration_rate:
                    # Exploration: Sample from a multivariate Gaussian distribution
                    z = np.random.multivariate_normal(np.zeros(self.dim), self.C)
                    new_population[i] = population[i] + z * (self.ub - self.lb) * 0.1
                    new_population[i] = np.clip(new_population[i], self.lb, self.ub)
                    self.velocities[i] = 0  # Reset velocity after exploration
                else:
                    # Exploitation: Guided by the best individual with velocity damping and constriction factor
                    cognitive_component = np.random.rand(self.dim) * (self.x_opt - population[i])
                    new_velocity = self.constriction_factor * (0.5 * self.velocities[i] + cognitive_component)
                    new_population[i] = population[i] + new_velocity
                    new_population[i] = np.clip(new_population[i], self.lb, self.ub)
                    self.velocities[i] = new_velocity # Update velocity

                # Local Search (Self-adaptive probability)
                if np.random.rand() < self.local_search_prob:
                    step_size = 0.01 * (self.ub - self.lb)
                    new_population[i] = population[i] + np.random.uniform(-step_size, step_size, size=self.dim)
                    new_population[i] = np.clip(new_population[i], self.lb, self.ub)
                    # Reduce local search probability
                    self.local_search_prob = max(self.local_search_prob * self.local_search_decay, self.min_local_search_prob)
                else:
                    # Increase local search probability slightly if not applied
                    self.local_search_prob = min(self.local_search_prob / self.local_search_decay, 0.1)

                new_fitness[i] = func(new_population[i]) # Evaluate inside the loop
                self.budget -= 1
                if self.budget <= 0:
                    break
            
            # Update population (replace if better)
            for i in range(self.pop_size):
                if new_fitness[i] < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]
                    self.success_counter += 1 # Increment success counter
                    
                    if new_fitness[i] < self.f_opt:
                        self.f_opt = new_fitness[i]
                        self.x_opt = new_population[i]

            # Update success rate
            self.success_rate = 0.8 * self.success_rate + 0.2 * (self.success_counter / self.pop_size)

            # Adapt covariance matrix (simplified)
            diff = population - np.mean(population, axis=0)
            self.C = (1 - self.learning_rate_C) * self.C + (self.learning_rate_C / self.pop_size) * np.dot(diff.T, diff)

            if self.budget <= 0:
                break

        return self.f_opt, self.x_opt