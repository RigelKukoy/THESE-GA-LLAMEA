import numpy as np
from scipy.stats import qmc

class AdaptivePopulationSearch:
    def __init__(self, budget=10000, dim=10, pop_size=20, exploration_rate=0.5, local_search_prob=0.1, stagnation_threshold=100, local_search_radius=0.05):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.exploration_rate = exploration_rate
        self.lb = -5.0
        self.ub = 5.0
        self.local_search_prob = local_search_prob
        self.stagnation_threshold = stagnation_threshold
        self.velocities = np.zeros((pop_size, dim))
        self.stagnation_counter = 0
        self.local_search_radius = local_search_radius * (self.ub - self.lb) # Radius as a fraction of the search space

        # Sobol sequence initialization for better diversity
        self.population = self._sobol_initialization(pop_size, dim)

    def _sobol_initialization(self, pop_size, dim):
        """Initialize population using Sobol sequence."""
        sampler = qmc.Sobol(d=dim, scramble=True)
        sample = sampler.random(n=pop_size)
        return self.lb + (self.ub - self.lb) * sample


    def __call__(self, func):
        population = self.population.copy() # Use initialized population
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size

        best_index = np.argmin(fitness)
        self.f_opt = fitness[best_index]
        self.x_opt = population[best_index]
        
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
                self.exploration_rate = min(self.exploration_rate + 0.2, 0.9) # Increase exploration
                self.stagnation_counter = 0 # Reset counter
            else:
                self.exploration_rate = max(self.exploration_rate - 0.05, 0.1) # Decrease exploration

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
                    new_population[i] = population[i] + new_velocity
                    new_population[i] = np.clip(new_population[i], self.lb, self.ub)
                    self.velocities[i] = new_velocity # Update velocity

                # Local Search (around the best so far)
                if np.random.rand() < self.local_search_prob:
                    # Sample from a normal distribution around the current best
                    new_population[i] = self.x_opt + np.random.normal(0, self.local_search_radius, size=self.dim)
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

        return self.f_opt, self.x_opt