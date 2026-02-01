import numpy as np

class RankDiversityPSO:
    def __init__(self, budget=10000, dim=10, pop_size=20, w=0.7, c1=1.5, c2=1.5, diversity_rate=0.1):
        """
        Initialize the Rank Diversity PSO algorithm.

        Args:
            budget (int): Total number of function evaluations.
            dim (int): Dimensionality of the problem.
            pop_size (int): Population size.
            w (float): Inertia weight for PSO.
            c1 (float): Cognitive coefficient for PSO.
            c2 (float): Social coefficient for PSO.
            diversity_rate (float): Probability of applying diversity maintenance.
        """
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.diversity_rate = diversity_rate
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

    def crowding_distance(self):
        """
        Calculate the crowding distance for each particle.
        """
        distances = np.zeros(self.pop_size)
        for m in range(self.dim):
            # Sort population based on the m-th dimension
            sorted_indices = np.argsort(self.pop[:, m])
            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf
            for i in range(1, self.pop_size - 1):
                distances[sorted_indices[i]] += (self.pop[sorted_indices[i+1], m] - self.pop[sorted_indices[i-1], m]) / (self.ub - self.lb)
        return distances

    def rank_based_velocity_update(self):
        """
        Update velocities based on fitness ranking.
        """
        ranked_indices = np.argsort(self.fitness)
        for i in range(self.pop_size):
            r1 = np.random.rand(self.dim)
            r2 = np.random.rand(self.dim)
            # Top 20% particles are considered "good"
            if i < int(0.2 * self.pop_size):  
                self.velocities[ranked_indices[i]] = self.w * self.velocities[ranked_indices[i]] + \
                                                     self.c1 * r1 * (self.x_best_global - self.pop[ranked_indices[i]]) + \
                                                     self.c2 * r2 * (self.pop[ranked_indices[0]] - self.pop[ranked_indices[i]]) # Attracted to best particle
            else:
                self.velocities[ranked_indices[i]] = self.w * self.velocities[ranked_indices[i]] + \
                                                     self.c1 * r1 * (self.x_best_global - self.pop[ranked_indices[i]]) + \
                                                     self.c2 * r2 * (self.pop[ranked_indices[int(0.2 * self.pop_size)]] - self.pop[ranked_indices[i]])  # Attracted to top 20%

            self.velocities[ranked_indices[i]] = np.clip(self.velocities[ranked_indices[i]], -self.v_max, self.v_max)

    def diversity_maintenance(self):
        """
        Maintain population diversity using crowding distance.
        """
        distances = self.crowding_distance()
        worst_particle_index = np.argmin(distances)
        self.pop[worst_particle_index] = np.random.uniform(self.lb, self.ub)
        self.fitness[worst_particle_index] = np.inf  # Mark as needing evaluation

    def pso_update(self, func):
        """
        Update the population using PSO principles with rank-based velocity update and diversity maintenance.

        Args:
            func: The black-box optimization function.
        """
        self.rank_based_velocity_update()

        for i in range(self.pop_size):
            new_position = self.pop[i] + self.velocities[i]
            new_position = np.clip(new_position, self.lb, self.ub)
            new_fitness = func(new_position)

            if new_fitness < self.fitness[i]:
                self.pop[i] = new_position
                self.fitness[i] = new_fitness
                if new_fitness < self.f_best_global:
                    self.f_best_global = new_fitness
                    self.x_best_global = self.pop[i].copy()

        if np.random.rand() < self.diversity_rate:
            self.diversity_maintenance()
            
            #Evaluate fitness of the newly generated particle after diversity maintenance
            distances = self.crowding_distance()
            worst_particle_index = np.argmin(distances)
            self.fitness[worst_particle_index] = func(self.pop[worst_particle_index])
            
            if self.fitness[worst_particle_index] < self.f_best_global:
                self.f_best_global = self.fitness[worst_particle_index]
                self.x_best_global = self.pop[worst_particle_index].copy()

    def __call__(self, func):
        """
        Optimize the given function using the Rank Diversity PSO algorithm.

        Args:
            func: The black-box optimization function.

        Returns:
            tuple: The best function value found and the corresponding solution.
        """
        self.initialize_population(func)
        eval_count = self.pop_size

        while eval_count < self.budget:
            self.w = 0.7 - 0.5 * (eval_count / self.budget)
            self.pso_update(func)
            eval_count += self.pop_size 
            
            if eval_count > self.budget:
                break

        return self.f_best_global, self.x_best_global