import numpy as np

class AdaptivePopulationSearch:
    def __init__(self, budget=10000, dim=10, initial_pop_size=20, exploration_rate=0.5, local_search_prob=0.1, stagnation_threshold=100, min_pop_size=5, max_pop_size=50):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = initial_pop_size
        self.pop_size = initial_pop_size
        self.exploration_rate = exploration_rate
        self.lb = -5.0
        self.ub = 5.0
        self.local_search_prob = local_search_prob
        self.stagnation_threshold = stagnation_threshold
        self.min_pop_size = min_pop_size
        self.max_pop_size = max_pop_size
        self.velocities = np.zeros((self.max_pop_size, dim)) # Allocate velocities for maximum pop size
        self.stagnation_counter = 0
        self.population = self._orthogonal_initialization(self.max_pop_size, dim) # Initialize up to the maximum pop size


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
        population = self.population[:self.initial_pop_size].copy() # Use initialized population
        fitness = np.array([func(x) for x in population])
        self.budget -= self.initial_pop_size
        self.pop_size = self.initial_pop_size

        best_index = np.argmin(fitness)
        self.f_opt = fitness[best_index]
        self.x_opt = population[best_index]
        
        previous_best_fitness = self.f_opt

        while self.budget > self.min_pop_size:
            # Stagnation detection
            if self.f_opt >= previous_best_fitness:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0

            previous_best_fitness = self.f_opt

            # Adaptive adjustment of population size based on remaining budget and stagnation
            if self.stagnation_counter > self.stagnation_threshold:
                self.pop_size = max(self.min_pop_size, self.pop_size // 2) # Reduce population size if stagnating
                self.exploration_rate = min(self.exploration_rate + 0.1, 0.9) # Increase exploration

                # Orthogonally re-initialize some individuals to add diversity
                num_reinitialized = min(self.pop_size // 2, self.pop_size)
                reinitialized_indices = np.random.choice(self.pop_size, num_reinitialized, replace=False)
                population[reinitialized_indices] = self._orthogonal_initialization(num_reinitialized, self.dim)
                fitness[reinitialized_indices] = [func(x) for x in population[reinitialized_indices]]  # Recalculate fitness
                self.budget -= num_reinitialized
                self.stagnation_counter = 0 # Reset counter
            else:
                self.pop_size = min(self.max_pop_size, self.pop_size * 2)  # Increase population size if improving
                if self.budget < self.pop_size:
                    self.pop_size = self.budget
                self.exploration_rate = max(self.exploration_rate - 0.02, 0.1) # Decrease exploration

            new_population = np.copy(population)
            for i in range(self.pop_size):
                if np.random.rand() < self.exploration_rate:
                    # Exploration: Randomly perturb the individual
                    new_population[i] = population[i] + np.random.uniform(-1.0, 1.0, size=self.dim) * (self.ub - self.lb) * 0.1
                    new_population[i] = np.clip(new_population[i], self.lb, self.ub)
                    self.velocities[i] = 0  # Reset velocity after exploration
                else:
                    # Exploitation: Guided by the best individual with velocity damping
                    inertia = 0.5
                    cognitive_component = np.random.rand(self.dim) * (self.x_opt - population[i])
                    new_velocity = inertia * self.velocities[i] + cognitive_component
                    
                    # Velocity clamping to prevent excessive jumps
                    max_velocity = 0.1 * (self.ub - self.lb)
                    new_velocity = np.clip(new_velocity, -max_velocity, max_velocity)
                    
                    new_population[i] = population[i] + new_velocity
                    new_population[i] = np.clip(new_population[i], self.lb, self.ub)
                    self.velocities[i] = new_velocity # Update velocity

                # Adaptive Local Search
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
                    
                    if new_fitness[i] < self.f_opt:
                        self.f_opt = new_fitness[i]
                        self.x_opt = new_population[i]
                        #Increase local search probability when finding better solution
                        self.local_search_prob = min(self.local_search_prob + 0.05, 0.5)
                else:
                    # Decrease local search probability when not finding better solutions
                    self.local_search_prob = max(self.local_search_prob - 0.01, 0.01)


        return self.f_opt, self.x_opt