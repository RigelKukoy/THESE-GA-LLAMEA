import numpy as np

class AdaptivePopulationSearch:
    def __init__(self, budget=10000, dim=10, pop_size=20, exploration_rate=0.5, local_search_prob=0.1, success_history_size=10):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.exploration_rate = exploration_rate
        self.lb = -5.0
        self.ub = 5.0
        self.local_search_prob = local_search_prob
        self.velocities = np.zeros((pop_size, dim)) # Initialize velocities for damping
        self.success_history = []
        self.success_history_size = success_history_size
        self.exploration_learning_rate = 0.1
        self.exploitation_learning_rate = 0.1
        self.min_pop_size = 5
        self.max_pop_size = 50

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
            # Adaptive adjustment of exploration rate based on success rate
            success_rate = np.mean(self.success_history) if self.success_history else 0.5

            # Adjust population size based on success rate
            if success_rate > 0.6 and self.pop_size < self.max_pop_size:
                self.pop_size = min(self.pop_size + 1, self.max_pop_size)
                population = np.vstack((population, np.random.uniform(self.lb, self.ub, size=(1, self.dim))))  # Add new individual
                self.velocities = np.vstack((self.velocities, np.zeros((1, self.dim))))
                fitness = np.append(fitness, np.inf) # Ensure the fitness is updated later
            elif success_rate < 0.4 and self.pop_size > self.min_pop_size:
                worst_index = np.argmax(fitness)
                population = np.delete(population, worst_index, axis=0)
                self.velocities = np.delete(self.velocities, worst_index, axis=0)
                fitness = np.delete(fitness, worst_index)
                self.pop_size = max(self.pop_size - 1, self.min_pop_size)

            new_population = np.copy(population)
            new_fitness = np.copy(fitness)
            success_count = 0
            for i in range(self.pop_size):
                if np.random.rand() < self.exploration_rate:
                    # Exploration: Randomly perturb the individual
                    step_size = self.exploration_learning_rate * (self.ub - self.lb)
                    new_population[i] = population[i] + np.random.uniform(-1.0, 1.0, size=self.dim) * step_size
                    new_population[i] = np.clip(new_population[i], self.lb, self.ub)
                    self.velocities[i] = 0  # Reset velocity after exploration
                else:
                    # Exploitation: Guided by the best individual with velocity damping
                    inertia = 0.5  # Inertia factor
                    cognitive_component = np.random.rand(self.dim) * (self.x_opt - population[i])
                    new_velocity = inertia * self.velocities[i] + self.exploitation_learning_rate * cognitive_component
                    new_population[i] = population[i] + new_velocity
                    new_population[i] = np.clip(new_population[i], self.lb, self.ub)
                    self.velocities[i] = new_velocity # Update velocity

                # Local Search
                if np.random.rand() < self.local_search_prob:
                    step_size = 0.01 * (self.ub - self.lb)
                    new_population[i] = population[i] + np.random.uniform(-step_size, step_size, size=self.dim)
                    new_population[i] = np.clip(new_population[i], self.lb, self.ub)

                # Evaluate fitness
                if self.budget > 0:
                    new_fitness_i = func(new_population[i])
                    self.budget -= 1
                else:
                    new_fitness_i = np.inf  # No more budget

                if new_fitness_i < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness_i
                    success_count += 1
                    if new_fitness_i < self.f_opt:
                        self.f_opt = new_fitness_i
                        self.x_opt = new_population[i]

            # Update exploration/exploitation rates based on success
            if success_count > 0:
                self.exploration_learning_rate *= 1.05
                self.exploitation_learning_rate *= 1.05
            else:
                self.exploration_learning_rate *= 0.95
                self.exploitation_learning_rate *= 0.95
            self.exploration_learning_rate = np.clip(self.exploration_learning_rate, 0.01, 0.5)
            self.exploitation_learning_rate = np.clip(self.exploitation_learning_rate, 0.01, 0.5)
            
            # Update success history
            self.success_history.append(success_count / self.pop_size)
            if len(self.success_history) > self.success_history_size:
                self.success_history.pop(0)

        return self.f_opt, self.x_opt