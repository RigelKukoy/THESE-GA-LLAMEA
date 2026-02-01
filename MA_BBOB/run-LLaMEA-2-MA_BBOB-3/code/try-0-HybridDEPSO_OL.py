import numpy as np
from scipy.stats import norm
from pyDOE import lhs  # For Latin Hypercube Sampling

class HybridDEPSO_OL:
    def __init__(self, budget=10000, dim=10, initial_pop_size=20, de_mutation_factor=0.5, pso_inertia=0.7, pso_cognitive=1.5, pso_social=1.5, ol_factor = 0.1):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = initial_pop_size
        self.pop_size = initial_pop_size
        self.population = np.random.uniform(-5, 5, size=(self.pop_size, dim))
        self.fitness = np.zeros(self.pop_size)
        self.velocities = np.zeros((self.pop_size, dim))
        self.best_positions = self.population.copy()
        self.best_fitness = np.full(self.pop_size, np.inf)
        self.global_best_position = None
        self.global_best_fitness = np.inf
        self.de_mutation_factor = de_mutation_factor
        self.pso_inertia = pso_inertia
        self.pso_cognitive = pso_cognitive
        self.pso_social = pso_social
        self.exploration_prob = 0.5  # Probability of exploration (DE/OL)
        self.exploitation_prob = 0.5  # Probability of exploitation (PSO)
        self.exploration_decay = 0.995 #Reduce exploration as budget is used.
        self.ol_factor = ol_factor # Factor for orthogonal learning.

    def orthogonal_learning(self, x, func):
        """
        Performs orthogonal learning based on the current solution x.
        """
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub

        # Generate a Latin Hypercube Sample around the current solution
        lhs_sample = lhs(dim, samples=dim)  # Generate dim samples
        orthogonal_points = np.zeros((dim, dim))

        for i in range(dim):
            orthogonal_points[i, :] = x + self.ol_factor * (lhs_sample[i, :] - 0.5) * (ub - lb) # Center around 0 and scale

        orthogonal_points = np.clip(orthogonal_points, lb, ub)
        
        # Evaluate the orthogonal points and select the best one
        best_orthogonal_fitness = np.inf
        best_orthogonal_point = None
        
        for i in range(dim):
            fitness = func(orthogonal_points[i, :])
            if fitness < best_orthogonal_fitness:
                best_orthogonal_fitness = fitness
                best_orthogonal_point = orthogonal_points[i, :].copy()
        
        return best_orthogonal_point, best_orthogonal_fitness


    def __call__(self, func):
        eval_count = 0

        # Initial evaluation
        for i in range(self.pop_size):
            if eval_count < self.budget:
                f = func(self.population[i])
                eval_count += 1
                self.fitness[i] = f
                if f < self.best_fitness[i]:
                    self.best_fitness[i] = f
                    self.best_positions[i] = self.population[i].copy()
                    if f < self.global_best_fitness:
                        self.global_best_fitness = f
                        self.global_best_position = self.population[i].copy()

        while eval_count < self.budget:
            for i in range(self.pop_size):
                if np.random.rand() < self.exploration_prob:
                    # Exploration (DE or Orthogonal Learning)
                    if np.random.rand() < 0.5:  # Choose between DE and OL
                        # Differential Evolution
                        r1, r2, r3 = np.random.choice(self.pop_size, 3, replace=False)
                        new_position = self.best_positions[r1] + self.de_mutation_factor * (self.population[r2] - self.population[r3])
                        new_position = np.clip(new_position, func.bounds.lb, func.bounds.ub)

                    else:
                        # Orthogonal Learning
                        new_position, new_fitness = self.orthogonal_learning(self.population[i], func)
                        if eval_count + self.dim > self.budget:
                            break  # Check budget before evaluating all orthogonal points
                        
                        
                else:
                    # Exploitation (PSO)
                    new_velocity = self.pso_inertia * self.velocities[i] + \
                                   self.pso_cognitive * np.random.rand(self.dim) * (self.best_positions[i] - self.population[i]) + \
                                   self.pso_social * np.random.rand(self.dim) * (self.global_best_position - self.population[i])
                    new_position = self.population[i] + new_velocity
                    new_position = np.clip(new_position, func.bounds.lb, func.bounds.ub)
                    self.velocities[i] = new_velocity

                # Evaluate new position
                f = func(new_position)
                eval_count += 1
                if eval_count >= self.budget:
                    break
                
                if f < self.fitness[i]:
                    self.population[i] = new_position
                    self.fitness[i] = f
                    if f < self.best_fitness[i]:
                        self.best_fitness[i] = f
                        self.best_positions[i] = self.population[i].copy()
                        if f < self.global_best_fitness:
                            self.global_best_fitness = f
                            self.global_best_position = self.population[i].copy()

            self.exploration_prob *= self.exploration_decay
            self.exploitation_prob = 1 - self.exploration_prob

        return self.global_best_fitness, self.global_best_position