import numpy as np

class HybridPSO_DE:
    def __init__(self, budget=10000, dim=10, pop_size=20, w=0.7, c1=1.5, c2=1.5, cr=0.7, f=0.8, v_max_ratio=0.2):
        """
        Initialize the Hybrid PSO-DE algorithm.

        Args:
            budget (int): Total number of function evaluations.
            dim (int): Dimensionality of the problem.
            pop_size (int): Population size.
            w (float): Inertia weight for PSO.
            c1 (float): Cognitive coefficient for PSO.
            c2 (float): Social coefficient for PSO.
            cr (float): Crossover rate for DE.
            f (float): Scaling factor for DE.
            v_max_ratio (float): Ratio to determine maximum velocity.
        """
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.cr = cr
        self.f = f
        self.v_max_ratio = v_max_ratio
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
        self.v_max = self.v_max_ratio * (self.ub - self.lb)

    def pso_update(self, func):
        """
        Update the population using PSO principles.

        Args:
            func: The black-box optimization function.
        """
        personal_best_locations = self.pop.copy()
        personal_best_fitness = self.fitness.copy()

        for i in range(self.pop_size):
            # Update velocities
            r1 = np.random.rand(self.dim)
            r2 = np.random.rand(self.dim)
            self.velocities[i] = self.w * self.velocities[i] + \
                                 self.c1 * r1 * (personal_best_locations[i] - self.pop[i]) + \
                                 self.c2 * r2 * (self.x_best_global - self.pop[i])

            # Velocity clamping
            self.velocities[i] = np.clip(self.velocities[i], -self.v_max, self.v_max)

            # Update positions
            self.pop[i] = self.pop[i] + self.velocities[i]

            # Boundary handling (clip to bounds)
            self.pop[i] = np.clip(self.pop[i], self.lb, self.ub)
            
            #Evaluate new positions
            new_fitness = func(self.pop[i])

            if new_fitness < self.fitness[i]:
                self.fitness[i] = new_fitness
                personal_best_locations[i] = self.pop[i].copy()

            if new_fitness < self.f_best_global:
                 self.f_best_global = new_fitness
                 self.x_best_global = self.pop[i].copy()

    def de_update(self, func):
        """
        Update the population using Differential Evolution principles.

        Args:
            func: The black-box optimization function.
        """
        for i in range(self.pop_size):
            # Choose three random indices, distinct from each other and i
            idxs = list(range(self.pop_size))
            idxs.remove(i)
            a, b, c = np.random.choice(idxs, 3, replace=False)

            # Mutation
            v = self.pop[a] + self.f * (self.pop[b] - self.pop[c])
            v = np.clip(v, self.lb, self.ub)

            # Crossover
            u = np.zeros(self.dim)
            j_rand = np.random.randint(0, self.dim)
            for j in range(self.dim):
                if np.random.rand() < self.cr or j == j_rand:
                    u[j] = v[j]
                else:
                    u[j] = self.pop[i, j]

            # Selection
            f_u = func(u)
            if f_u < self.fitness[i]:
                self.pop[i] = u
                self.fitness[i] = f_u

                if f_u < self.f_best_global:
                    self.f_best_global = f_u
                    self.x_best_global = u.copy()


    def __call__(self, func):
        """
        Optimize the given function using the Hybrid PSO-DE algorithm.

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
            
            #Alternate between PSO and DE updates
            if eval_count % 2 == 0:
                self.pso_update(func)
            else:
                self.de_update(func)
            
            eval_count = np.sum([1 for i in range(self.pop_size) if i in range(self.pop_size)])+ self.pop_size #Simple way to account for the budget.

            eval_count = np.sum([1 for i in range(self.pop_size) if i in range(self.pop_size)]) + self.pop_size


            if eval_count > self.budget:
                break

        return self.f_best_global, self.x_best_global