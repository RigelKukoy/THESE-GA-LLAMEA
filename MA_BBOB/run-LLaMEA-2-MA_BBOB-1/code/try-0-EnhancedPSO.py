import numpy as np

class EnhancedPSO:
    def __init__(self, budget=10000, dim=10, pop_size=20, w=0.7, c1=1.5, c2=1.5, mutation_rate=0.1, exploration_rate=0.3):
        """
        Initialize the Enhanced PSO algorithm.

        Args:
            budget (int): Total number of function evaluations.
            dim (int): Dimensionality of the problem.
            pop_size (int): Population size.
            w (float): Inertia weight for PSO.
            c1 (float): Cognitive coefficient for PSO.
            c2 (float): Social coefficient for PSO.
            mutation_rate (float): Probability of applying velocity mutation.
            exploration_rate (float): Probability of random exploration.
        """
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.mutation_rate = mutation_rate
        self.exploration_rate = exploration_rate
        self.pop = None
        self.fitness = None
        self.velocities = None
        self.x_best_global = None
        self.f_best_global = np.inf
        self.lb = None
        self.ub = None
        self.v_max = None

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


    def velocity_mutation(self, i):
        """
        Apply differential evolution-inspired velocity mutation.

        Args:
            i (int): Index of the particle to mutate.
        """
        idxs = np.random.choice(self.pop_size, 3, replace=False)
        while i in idxs:
            idxs = np.random.choice(self.pop_size, 3, replace=False)

        F = 0.8  # Mutation factor
        self.velocities[i] += F * (self.velocities[idxs[0]] - self.velocities[idxs[1]]) + F * (self.velocities[idxs[2]] - self.velocities[i])

    def pso_update(self, func):
        """
        Update the population using PSO principles with velocity mutation and adaptive exploration.

        Args:
            func: The black-box optimization function.
        """
        for i in range(self.pop_size):
            # Update velocities
            r1 = np.random.rand(self.dim)
            r2 = np.random.rand(self.dim)
            self.velocities[i] = self.w * self.velocities[i] + \
                                 self.c1 * r1 * (self.pop[i] - self.pop[i]) + \
                                 self.c2 * r2 * (self.x_best_global - self.pop[i])

            # Velocity clamping
            self.velocities[i] = np.clip(self.velocities[i], -self.v_max, self.v_max)

            # Velocity mutation
            if np.random.rand() < self.mutation_rate:
                self.velocity_mutation(i)

            # Update positions
            new_position = self.pop[i] + self.velocities[i]

            # Boundary handling (clip to bounds)
            new_position = np.clip(new_position, self.lb, self.ub)

            # Adaptive exploration: replace particle with random position
            if np.random.rand() < self.exploration_rate:
                new_position = np.random.uniform(self.lb, self.ub)

            new_fitness = func(new_position)  # Evaluate new position
            
            if new_fitness < self.fitness[i]:
                self.pop[i] = new_position
                self.fitness[i] = new_fitness
                if new_fitness < self.f_best_global:
                    self.f_best_global = new_fitness
                    self.x_best_global = self.pop[i].copy()
        
    def __call__(self, func):
        """
        Optimize the given function using the Enhanced PSO algorithm.

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