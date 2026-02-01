import numpy as np

class CauchyPSO:
    def __init__(self, budget=10000, dim=10, pop_size=20, w=0.7, c1=1.5, c2=1.5, v_max_ratio=0.2, cauchy_scale=1.0, restart_freq=500):
        """
        Initialize the Cauchy PSO algorithm.

        Args:
            budget (int): Total number of function evaluations.
            dim (int): Dimensionality of the problem.
            pop_size (int): Population size.
            w (float): Inertia weight for PSO.
            c1 (float): Cognitive coefficient for PSO.
            c2 (float): Social coefficient for PSO.
            v_max_ratio (float): Ratio to determine maximum velocity.
            cauchy_scale (float): Scale parameter for Cauchy mutation.
            restart_freq (int): Frequency of restarting the population.
        """
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.v_max_ratio = v_max_ratio
        self.cauchy_scale = cauchy_scale
        self.restart_freq = restart_freq
        self.pop = None
        self.fitness = None
        self.velocities = None
        self.x_best_global = None
        self.f_best_global = np.inf
        self.lb = None
        self.ub = None
        self.v_max = None
        self.eval_count = 0

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
        self.eval_count += self.pop_size

    def cauchy_mutation(self):
        """
        Apply Cauchy mutation to a randomly selected particle.
        """
        i = np.random.randint(self.pop_size)
        mutation = self.cauchy_scale * np.random.standard_cauchy(size=self.dim)
        self.pop[i] = np.clip(self.pop[i] + mutation, self.lb, self.ub)
        return i # Return mutated index to update fitness

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
            f = func(self.pop[i])
            self.eval_count +=1

            if f < self.fitness[i]:
                self.fitness[i] = f
                personal_best_locations[i] = self.pop[i].copy()

            if f < self.f_best_global:
                 self.f_best_global = f
                 self.x_best_global = self.pop[i].copy()

    def restart_population(self, func):
        """
        Restart the population with new random solutions.
        """
        self.pop = np.random.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.pop])
        self.x_best_global = self.pop[np.argmin(self.fitness)].copy()
        self.f_best_global = np.min(self.fitness)
        self.eval_count += self.pop_size

    def __call__(self, func):
        """
        Optimize the given function using the Cauchy PSO algorithm.

        Args:
            func: The black-box optimization function.

        Returns:
            tuple: The best function value found and the corresponding solution.
        """
        self.initialize_population(func)

        while self.eval_count < self.budget:
            # PSO update
            self.pso_update(func)

            # Cauchy mutation (exploration)
            if self.eval_count < self.budget:
                mutated_idx = self.cauchy_mutation()
                f_mutated = func(self.pop[mutated_idx])
                self.eval_count += 1
                self.fitness[mutated_idx] = f_mutated
                if f_mutated < self.f_best_global:
                    self.f_best_global = f_mutated
                    self.x_best_global = self.pop[mutated_idx].copy()

            # Restart mechanism
            if self.eval_count > 0 and self.eval_count % self.restart_freq == 0 and self.eval_count < self.budget:
                self.restart_population(func)

        return self.f_best_global, self.x_best_global