import numpy as np

class AdaptivePopulationSearch:
    def __init__(self, budget=10000, dim=10, pop_size=20, exploration_rate=0.5, local_search_prob=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.exploration_rate = exploration_rate
        self.lb = -5.0
        self.ub = 5.0
        self.local_search_prob = local_search_prob
        self.velocities = np.zeros((pop_size, dim)) # Initialize velocities for damping

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size

        # Find best individual
        best_index = np.argmin(fitness)
        self.f_opt = fitness[best_index]
        self.x_opt = population[best_index]

        # Main loop
        while self.budget > 0:
            # Adaptive adjustment of exploration rate based on fitness variance
            fitness_variance = np.var(fitness)
            if fitness_variance > 1e-3: # If variance is high, explore more
                exploration_rate = min(self.exploration_rate + 0.1, 0.9)
            else: # If variance is low, exploit more
                exploration_rate = max(self.exploration_rate - 0.1, 0.1)

            new_population = np.copy(population)
            for i in range(self.pop_size):
                if np.random.rand() < exploration_rate:
                    # Exploration: Randomly perturb the individual
                    new_population[i] = population[i] + np.random.uniform(-1.0, 1.0, size=self.dim) * (self.ub - self.lb) * 0.1
                    new_population[i] = np.clip(new_population[i], self.lb, self.ub)
                    self.velocities[i] = 0  # Reset velocity after exploration
                else:
                    # Exploitation: Guided by the best individual with velocity damping
                    inertia = 0.5  # Inertia factor
                    cognitive_component = np.random.rand(self.dim) * (self.x_opt - population[i])
                    new_velocity = inertia * self.velocities[i] + cognitive_component
                    new_population[i] = population[i] + new_velocity
                    new_population[i] = np.clip(new_population[i], self.lb, self.ub)
                    self.velocities[i] = new_velocity # Update velocity

                # Local Search
                if np.random.rand() < self.local_search_prob:
                    step_size = 0.01 * (self.ub - self.lb)
                    new_population[i] = population[i] + np.random.uniform(-step_size, step_size, size=self.dim)
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