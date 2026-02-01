import numpy as np

class AdaptiveHybridPSO_LS:
    def __init__(self, budget=10000, dim=10, pop_size=20, w=0.7, c1=1.5, c2=1.5, ls_prob=0.1, step_size=0.1):
        """
        Initialize the Adaptive Hybrid PSO-LS algorithm.

        Args:
            budget (int): Total number of function evaluations.
            dim (int): Dimensionality of the problem.
            pop_size (int): Population size.
            w (float): Inertia weight for PSO.
            c1 (float): Cognitive coefficient for PSO.
            c2 (float): Social coefficient for PSO.
            ls_prob (float): Probability of applying local search to a particle.
            step_size (float): Step size for the local search.
        """
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.ls_prob = ls_prob
        self.step_size = step_size
        self.pop = None
        self.fitness = None
        self.velocities = None
        self.x_best_global = None
        self.f_best_global = np.inf
        self.lb = None
        self.ub = None
        self.eval_count = 0
        self.pso_frequency = 0.5 # Initial frequency for PSO

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
        self.eval_count += self.pop_size

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

            # Update positions
            self.pop[i] = self.pop[i] + self.velocities[i]

            # Boundary handling (clip to bounds)
            self.pop[i] = np.clip(self.pop[i], self.lb, self.ub)
            
            #Evaluate new positions
            new_fitness = func(self.pop[i])
            self.eval_count += 1

            if new_fitness < self.fitness[i]:
                self.fitness[i] = new_fitness
                personal_best_locations[i] = self.pop[i].copy()

            if new_fitness < self.f_best_global:
                 self.f_best_global = new_fitness
                 self.x_best_global = self.pop[i].copy()

    def local_search(self, func, x):
        """
        Perform a simple gradient-based local search.

        Args:
            func: The black-box optimization function.
            x (np.ndarray): Starting point for the local search.

        Returns:
            tuple: Improved solution and its fitness.
        """
        x_current = x.copy()
        f_current = func(x_current)
        self.eval_count += 1
        
        for _ in range(5): # Limited iterations for budget concerns
            # Generate a random direction
            direction = np.random.uniform(-1, 1, size=self.dim)
            direction = direction / np.linalg.norm(direction)  # Normalize

            # Take a step in that direction
            x_new = x_current + self.step_size * direction
            x_new = np.clip(x_new, self.lb, self.ub)  # Boundary handling
            
            f_new = func(x_new)
            self.eval_count += 1

            if f_new < f_current:
                x_current = x_new
                f_current = f_new

        return x_current, f_current

    def __call__(self, func):
        """
        Optimize the given function using the Adaptive Hybrid PSO-LS algorithm.

        Args:
            func: The black-box optimization function.

        Returns:
            tuple: The best function value found and the corresponding solution.
        """
        self.initialize_population(func)

        last_best_fitness = self.f_best_global
        pso_success = 0
        ls_success = 0

        while self.eval_count < self.budget:
            # Adaptive frequency adjustment
            if self.pso_frequency < 0.9 and self.eval_count % self.pop_size == 0:
                if self.f_best_global < last_best_fitness:
                  pso_success += 1
                last_best_fitness = self.f_best_global
            if self.pso_frequency > 0.1 and self.eval_count % self.pop_size == 0:
                if self.f_best_global < last_best_fitness:
                  ls_success += 1
                last_best_fitness = self.f_best_global

            if np.random.rand() < self.pso_frequency:
                self.pso_update(func)
            else:
                for i in range(self.pop_size):
                    if np.random.rand() < self.ls_prob:
                        x_new, f_new = self.local_search(func, self.pop[i])
                        if f_new < self.fitness[i]:
                            self.pop[i] = x_new
                            self.fitness[i] = f_new
                            if f_new < self.f_best_global:
                                self.f_best_global = f_new
                                self.x_best_global = x_new.copy()

            # Adaptive adjustment of PSO frequency
            if pso_success > ls_success:
                self.pso_frequency = min(self.pso_frequency + 0.05, 0.9)
            else:
                self.pso_frequency = max(self.pso_frequency - 0.05, 0.1)
            
            pso_success = 0
            ls_success = 0
            last_best_fitness = self.f_best_global


        return self.f_best_global, self.x_best_global