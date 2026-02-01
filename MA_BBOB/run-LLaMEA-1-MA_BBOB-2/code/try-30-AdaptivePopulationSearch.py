import numpy as np

class AdaptivePopulationSearch:
    def __init__(self, budget=10000, dim=10, pop_size=20, exploration_rate=0.5, local_search_init_prob=0.1, stagnation_threshold=100, momentum=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.exploration_rate = exploration_rate
        self.lb = -5.0
        self.ub = 5.0
        self.local_search_init_prob = local_search_init_prob
        self.local_search_prob = local_search_init_prob
        self.stagnation_threshold = stagnation_threshold
        self.momentum = momentum
        self.velocities = np.zeros((pop_size, dim))
        self.stagnation_counter = 0
        self.success_counter = 0
        self.population = self._orthogonal_initialization(pop_size, dim)

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

        while self.budget > 0:
            # Stagnation detection
            if self.f_opt >= previous_best_fitness:
                self.stagnation_counter += 1
                self.success_counter = 0
            else:
                self.stagnation_counter = 0
                self.success_counter +=1

            previous_best_fitness = self.f_opt

            # Adaptive adjustment of exploration rate based on stagnation
            if self.stagnation_counter > self.stagnation_threshold:
                self.exploration_rate = min(self.exploration_rate + 0.2, 0.9) # Increase exploration
                self.stagnation_counter = 0 # Reset counter
                # Increase pop size if stagnating
                self.pop_size = min(self.pop_size + 5, 50)
            else:
                self.exploration_rate = max(self.exploration_rate - 0.05, 0.1) # Decrease exploration
                 # Reduce pop size if improving
                self.pop_size = max(self.pop_size - 1, 10)

            # Adjust local search probability based on success
            if self.success_counter > 5:
                self.local_search_prob = min(self.local_search_prob + 0.02, 0.5)
            else:
                 self.local_search_prob = max(self.local_search_prob - 0.01, 0.01)

            new_population = np.copy(population)
            new_velocities = np.copy(self.velocities)
            for i in range(self.pop_size):
                if np.random.rand() < self.exploration_rate:
                    # Exploration: Randomly perturb the individual
                    new_population[i] = population[i] + np.random.uniform(-1.0, 1.0, size=self.dim) * (self.ub - self.lb) * 0.1
                    new_population[i] = np.clip(new_population[i], self.lb, self.ub)
                    new_velocities[i] = 0  # Reset velocity after exploration
                else:
                    # Exploitation: Guided by the best individual with momentum
                    cognitive_component = np.random.rand(self.dim) * (self.x_opt - population[i])
                    new_velocity = self.momentum * self.velocities[i] + cognitive_component
                    new_population[i] = population[i] + new_velocity
                    new_population[i] = np.clip(new_population[i], self.lb, self.ub)
                    new_velocities[i] = new_velocity # Update velocity

                # Local Search
                if np.random.rand() < self.local_search_prob:
                    step_size = 0.01 * (self.ub - self.lb)
                    new_population[i] = population[i] + np.random.uniform(-step_size, step_size, size=self.dim)
                    new_population[i] = np.clip(new_population[i], self.lb, self.ub)

            new_fitness = np.array([func(x) for x in new_population])
            self.budget -= self.pop_size
            
            # Update population (replace if better)
            for i in range(self.pop_size):
                if new_fitness[i] < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]
                    self.velocities[i] = new_velocities[i]
                    
                    if new_fitness[i] < self.f_opt:
                        self.f_opt = new_fitness[i]
                        self.x_opt = new_population[i]

        return self.f_opt, self.x_opt