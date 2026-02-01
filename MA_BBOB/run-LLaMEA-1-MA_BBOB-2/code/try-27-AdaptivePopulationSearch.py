import numpy as np

class AdaptivePopulationSearch:
    def __init__(self, budget=10000, dim=10, pop_size=20, exploration_rate=0.5, local_search_prob=0.1, age_limit=5):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.exploration_rate = exploration_rate
        self.lb = -5.0
        self.ub = 5.0
        self.local_search_prob = local_search_prob
        self.velocities = np.zeros((pop_size, dim)) # Initialize velocities for damping
        self.ages = np.zeros(pop_size, dtype=int) # Initialize ages for each individual
        self.age_limit = age_limit

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
            new_fitness = np.copy(fitness)
            for i in range(self.pop_size):
                if np.random.rand() < exploration_rate or self.ages[i] >= self.age_limit:
                    # Exploration: Randomly perturb the individual or re-initialize old individuals
                    new_population[i] = np.random.uniform(self.lb, self.ub, size=self.dim)
                    self.velocities[i] = 0  # Reset velocity after exploration
                    self.ages[i] = 0 # Reset age
                else:
                    # Exploitation: Guided by the best individual with velocity damping
                    inertia = 0.5  # Inertia factor
                    cognitive_component = np.random.rand(self.dim) * (self.x_opt - population[i])
                    
                    # Dynamic velocity damping
                    damping_factor = 1.0 - (self.ages[i] / self.age_limit) if self.age_limit > 0 else 1.0
                    new_velocity = inertia * self.velocities[i] * damping_factor + cognitive_component
                    
                    new_population[i] = population[i] + new_velocity
                    new_population[i] = np.clip(new_population[i], self.lb, self.ub)
                    self.velocities[i] = new_velocity # Update velocity

                # Enhanced Local Search: Adaptive step size
                if np.random.rand() < self.local_search_prob:
                    step_size = 0.1 * (self.ub - self.lb) * np.exp(-self.ages[i])  # Reduce step size with age
                    local_search_direction = np.random.uniform(-1.0, 1.0, size=self.dim)
                    new_population[i] = population[i] + step_size * local_search_direction
                    new_population[i] = np.clip(new_population[i], self.lb, self.ub)

                # Evaluate fitness
                if self.budget > 0:
                    new_fitness[i] = func(new_population[i])
                    self.budget -= 1
                else:
                    new_fitness[i] = fitness[i] # Keep old fitness if budget is exceeded

            # Update population (replace if better)
            for i in range(self.pop_size):
                if new_fitness[i] < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]
                    self.ages[i] = 0 # Reset age if improved
                    
                    if new_fitness[i] < self.f_opt:
                        self.f_opt = new_fitness[i]
                        self.x_opt = new_population[i]
                else:
                    self.ages[i] += 1 # Increment age if not improved

        return self.f_opt, self.x_opt