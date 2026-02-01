import numpy as np

class EnhancedAdaptivePopulationSearch:
    def __init__(self, budget=10000, dim=10, pop_size=20, exploration_rate=0.5, local_search_prob=0.1, stagnation_threshold=100, constriction_factor=0.729, memory_size=10):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.exploration_rate = exploration_rate
        self.lb = -5.0
        self.ub = 5.0
        self.local_search_prob = local_search_prob
        self.stagnation_threshold = stagnation_threshold
        self.constriction_factor = constriction_factor
        self.memory_size = memory_size
        self.velocities = np.zeros((pop_size, dim))
        self.stagnation_counter = 0
        self.local_search_decay = 0.99
        self.min_local_search_prob = 0.01
        self.position_memory = np.zeros((pop_size, memory_size, dim))
        self.fitness_memory = np.full((pop_size, memory_size), np.inf)
        self.best_known_position = np.zeros((pop_size, dim))
        self.best_known_fitness = np.full(pop_size, np.inf)
        self.adaptive_constriction_factors = np.full(pop_size, constriction_factor)


        # Orthogonal initialization for better diversity
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

    def _local_search(self, x, func, step_size):
        """More sophisticated local search using multiple perturbations."""
        best_x = x
        best_fitness = func(x)
        self.budget -= 1

        for _ in range(5):  # Multiple perturbations
            new_x = x + np.random.uniform(-step_size, step_size, size=self.dim)
            new_x = np.clip(new_x, self.lb, self.ub)
            new_fitness = func(new_x)
            self.budget -= 1

            if new_fitness < best_fitness:
                best_fitness = new_fitness
                best_x = new_x

            if self.budget <= 0:
                break
        return best_x, best_fitness

    def __call__(self, func):
        population = self.population.copy()
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size

        best_index = np.argmin(fitness)
        self.f_opt = fitness[best_index]
        self.x_opt = population[best_index]

        self.best_known_position = population.copy()
        self.best_known_fitness = fitness.copy()

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
                self.exploration_rate = min(self.exploration_rate + 0.2, 0.9)
                self.stagnation_counter = 0
            else:
                self.exploration_rate = max(self.exploration_rate - 0.05, 0.1)

            new_population = np.copy(population)
            for i in range(self.pop_size):
                # Memory update
                self.position_memory[i, 0] = population[i]
                self.fitness_memory[i, 0] = fitness[i]

                # Shift memory
                self.position_memory[i, 1:] = self.position_memory[i, :-1].copy()
                self.fitness_memory[i, 1:] = self.fitness_memory[i, :-1].copy()

                if np.random.rand() < self.exploration_rate:
                    # Exploration: Randomly perturb the individual
                    new_population[i] = population[i] + np.random.uniform(-1.0, 1.0, size=self.dim) * (self.ub - self.lb) * 0.1
                    new_population[i] = np.clip(new_population[i], self.lb, self.ub)
                    self.velocities[i] = 0  # Reset velocity after exploration
                    self.adaptive_constriction_factors[i] = self.constriction_factor
                else:
                    # Exploitation: Guided by the best individual and memory
                    cognitive_component = np.random.rand(self.dim) * (self.best_known_position[i] - population[i])

                    # Memory component: biased towards best memory
                    memory_index = np.argmin(self.fitness_memory[i])
                    memory_component = np.random.rand(self.dim) * (self.position_memory[i, memory_index] - population[i])

                    new_velocity = self.adaptive_constriction_factors[i] * (0.5 * self.velocities[i] + cognitive_component + 0.2 * memory_component)
                    new_population[i] = population[i] + new_velocity
                    new_population[i] = np.clip(new_population[i], self.lb, self.ub)
                    self.velocities[i] = new_velocity # Update velocity
                    
                    # Adaptive constriction factor
                    if fitness[i] < self.best_known_fitness[i]:
                        self.adaptive_constriction_factors[i] = min(self.adaptive_constriction_factors[i] + 0.01, 0.9)
                    else:
                        self.adaptive_constriction_factors[i] = max(self.adaptive_constriction_factors[i] - 0.01, 0.5)

                # Local Search (Self-adaptive probability)
                if np.random.rand() < self.local_search_prob and self.budget > 0:
                    step_size = 0.01 * (self.ub - self.lb)
                    new_population[i], new_fitness_local = self._local_search(new_population[i], func, step_size)
                    if self.budget <= 0:
                        break
                    # Reduce local search probability
                    self.local_search_prob = max(self.local_search_prob * self.local_search_decay, self.min_local_search_prob)
                else:
                    # Increase local search probability slightly if not applied
                    self.local_search_prob = min(self.local_search_prob / self.local_search_decay, 0.1)

            new_fitness = np.array([func(x) for x in new_population])
            self.budget -= self.pop_size

            # Update population (replace if better)
            for i in range(self.pop_size):
                if new_fitness[i] < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]

                    # Update best known position
                    if new_fitness[i] < self.best_known_fitness[i]:
                        self.best_known_fitness[i] = new_fitness[i]
                        self.best_known_position[i] = new_population[i]
                    
                    if new_fitness[i] < self.f_opt:
                        self.f_opt = new_fitness[i]
                        self.x_opt = new_population[i]
            
            if self.budget <= 0:
                break

        return self.f_opt, self.x_opt