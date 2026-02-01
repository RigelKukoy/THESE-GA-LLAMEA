import numpy as np

class SelfOrganizingPSO:
    def __init__(self, budget=10000, dim=10, pop_size=20, w=0.7, c1=1.5, c2=1.5, neighborhood_size=3, scout_rate=0.1):
        """
        Initialize the Self-Organizing PSO algorithm.

        Args:
            budget (int): Total number of function evaluations.
            dim (int): Dimensionality of the problem.
            pop_size (int): Population size.
            w (float): Inertia weight for PSO.
            c1 (float): Cognitive coefficient for PSO.
            c2 (float): Social coefficient for PSO.
            neighborhood_size (int): Size of the neighborhood for local best.
            scout_rate (float): Probability of a particle becoming a scout.
        """
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.neighborhood_size = neighborhood_size
        self.scout_rate = scout_rate
        self.pop = None
        self.fitness = None
        self.velocities = None
        self.x_best_global = None
        self.f_best_global = np.inf
        self.lb = None
        self.ub = None
        self.v_max = None
        self.x_best_local = None  # Local best for each particle

    def initialize_population(self, func):
        """
        Initialize the population within the bounds of the function.

        Args:
            func: The black-box optimization function.
        """
        self.lb = func.bounds.lb
        self.ub = func.bounds.ub
        self.pop = np.random.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.pop])
        self.velocities = np.zeros((self.pop_size, self.dim))
        self.x_best_global = self.pop[np.argmin(self.fitness)].copy()
        self.f_best_global = np.min(self.fitness)
        self.v_max = 0.2 * (self.ub - self.lb)
        self.x_best_local = self.pop.copy()  # Initially, local best is the particle itself

    def neighborhood_mutation(self, i):
        """
        Apply mutation based on the neighborhood's best.

        Args:
            i (int): Index of the particle to mutate.
        """
        # Find neighborhood indices
        neighborhood_indices = [(i + j) % self.pop_size for j in range(-self.neighborhood_size // 2, self.neighborhood_size // 2 + 1)]

        # Find the best particle within the neighborhood
        neighborhood_fitness = self.fitness[neighborhood_indices]
        best_neighbor_index = neighborhood_indices[np.argmin(neighborhood_fitness)]
        x_best_neighbor = self.pop[best_neighbor_index]
        
        # Mutate the particle's velocity towards the neighborhood best
        mutation_strength = 0.5  # Adjust as needed
        self.velocities[i] += mutation_strength * (x_best_neighbor - self.pop[i])

    def scout_movement(self, i):
        """
        Replace a particle with a random scout particle.

        Args:
            i (int): Index of the particle to replace.
        """
        self.pop[i] = np.random.uniform(self.lb, self.ub)
        self.velocities[i] = np.zeros(self.dim) # Reset velocity
    
    def pso_update(self, func):
        """
        Update the population using PSO principles with neighborhood mutation and self-organizing scouts.

        Args:
            func: The black-box optimization function.
        """
        for i in range(self.pop_size):
            # Update velocities
            r1 = np.random.rand(self.dim)
            r2 = np.random.rand(self.dim)
            self.velocities[i] = self.w * self.velocities[i] + \
                                 self.c1 * r1 * (self.x_best_local[i] - self.pop[i]) + \
                                 self.c2 * r2 * (self.x_best_global - self.pop[i])

            # Velocity clamping
            self.velocities[i] = np.clip(self.velocities[i], -self.v_max, self.v_max)

            # Neighborhood-based mutation
            self.neighborhood_mutation(i)

            # Update positions
            new_position = self.pop[i] + self.velocities[i]

            # Boundary handling (clip to bounds)
            new_position = np.clip(new_position, self.lb, self.ub)

            # Self-organizing scout behavior
            if np.random.rand() < self.scout_rate:
                self.scout_movement(i)
                new_position = self.pop[i] # scout_movement already updates position

            new_fitness = func(new_position)  # Evaluate new position

            if new_fitness < self.fitness[i]:
                self.pop[i] = new_position
                self.fitness[i] = new_fitness
                self.x_best_local[i] = new_position.copy() #Update local best
                if new_fitness < self.f_best_global:
                    self.f_best_global = new_fitness
                    self.x_best_global = self.pop[i].copy()

    def __call__(self, func):
        """
        Optimize the given function using the Self-Organizing PSO algorithm.

        Args:
            func: The black-box optimization function.

        Returns:
            tuple: The best function value found and the corresponding solution.
        """
        self.initialize_population(func)
        eval_count = self.pop_size # Account for initial population evaluation

        while eval_count < self.budget:
            # Adaptive parameter adjustment (example: linear decrease of inertia weight)
            self.w = 0.7 - 0.5 * (eval_count / self.budget)
            self.pso_update(func)
            eval_count += self.pop_size # Account for population evaluation

            if eval_count > self.budget:
                break

        return self.f_best_global, self.x_best_global