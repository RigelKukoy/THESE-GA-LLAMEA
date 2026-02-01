import numpy as np

class PSO_SA:
    def __init__(self, budget=10000, dim=10, pop_size=20, w=0.7, c1=1.5, c2=1.5, temp_init=1.0, temp_decay=0.99, v_max_ratio=0.2):
        """
        Initialize the PSO with Simulated Annealing algorithm.

        Args:
            budget (int): Total number of function evaluations.
            dim (int): Dimensionality of the problem.
            pop_size (int): Population size.
            w (float): Inertia weight for PSO.
            c1 (float): Cognitive coefficient for PSO.
            c2 (float): Social coefficient for PSO.
            temp_init (float): Initial temperature for SA.
            temp_decay (float): Temperature decay rate for SA.
            v_max_ratio (float): Ratio to determine maximum velocity.
        """
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.temp_init = temp_init
        self.temp_decay = temp_decay
        self.v_max_ratio = v_max_ratio
        self.pop = None
        self.fitness = None
        self.velocities = None
        self.x_best_global = None
        self.f_best_global = np.inf
        self.lb = None
        self.ub = None
        self.v_max = None
        self.temp = temp_init

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
        Update the population using PSO principles with SA acceptance.

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
            new_position = self.pop[i] + self.velocities[i]

            # Boundary handling (clip to bounds)
            new_position = np.clip(new_position, self.lb, self.ub)
            
            #Evaluate new positions
            new_fitness = func(new_position)

            # Simulated Annealing acceptance criterion
            delta_e = new_fitness - self.fitness[i]
            if delta_e < 0 or np.random.rand() < np.exp(-delta_e / self.temp):
                self.pop[i] = new_position
                self.fitness[i] = new_fitness
                if new_fitness < personal_best_fitness[i]:
                    personal_best_locations[i] = self.pop[i].copy()
                    personal_best_fitness[i] = new_fitness

                if new_fitness < self.f_best_global:
                    self.f_best_global = new_fitness
                    self.x_best_global = self.pop[i].copy()
        
        self.temp *= self.temp_decay #Cooling

    def __call__(self, func):
        """
        Optimize the given function using the PSO-SA algorithm.

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