import numpy as np

class HybridSAMetaheuristic:
    def __init__(self, budget=10000, dim=10, initial_pop_size=20, initial_temp=100.0, cooling_rate=0.95, de_mutation_factor=0.5, pso_inertia=0.7):
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
        self.initial_temp = initial_temp
        self.temperature = initial_temp
        self.cooling_rate = cooling_rate
        self.de_mutation_factor = de_mutation_factor
        self.pso_inertia = pso_inertia
        self.adaptation_rate = 0.1
        self.min_temp = 1e-4

    def __call__(self, func):
        eval_count = 0
        while eval_count < self.budget:
            # Evaluate fitness
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

            # Adaptive Strategy Selection & Parameter Adjustment
            for i in range(self.pop_size):
                # Blend DE and PSO
                r1, r2, r3 = np.random.choice(self.pop_size, 3, replace=False)
                de_vector = self.best_positions[r1] + self.de_mutation_factor * (self.population[r2] - self.population[r3])
                pso_velocity = self.pso_inertia * self.velocities[i] + \
                               2.0 * np.random.rand(self.dim) * (self.best_positions[i] - self.population[i]) + \
                               2.0 * np.random.rand(self.dim) * (self.global_best_position - self.population[i])

                new_position = self.population[i] + 0.5 * (de_vector - self.population[i]) + 0.5 * pso_velocity
                new_position = np.clip(new_position, func.bounds.lb, func.bounds.ub)

                # Simulated Annealing acceptance criterion
                delta_e = func(new_position) - self.fitness[i] # Evaluate new position here.
                eval_count += 1
                if eval_count >= self.budget:
                  break
                
                if delta_e < 0 or np.random.rand() < np.exp(-delta_e / self.temperature):
                    self.population[i] = new_position
                    self.fitness[i] = func(self.population[i]) # Evaluate new position.
                    eval_count += 1
                    if eval_count >= self.budget:
                      break
                    if self.fitness[i] < self.best_fitness[i]:
                        self.best_fitness[i] = self.fitness[i]
                        self.best_positions[i] = self.population[i].copy()
                        if self.fitness[i] < self.global_best_fitness:
                            self.global_best_fitness = self.fitness[i]
                            self.global_best_position = self.population[i].copy()
                
                self.velocities[i] = pso_velocity

            # Temperature Cooling
            self.temperature *= self.cooling_rate
            self.temperature = max(self.temperature, self.min_temp)

            # Population Size Adjustment (Adaptive)
            improvement_rate = np.sum(self.fitness > self.best_fitness) / self.pop_size
            if improvement_rate > 0.2 and self.pop_size < 2 * self.initial_pop_size:
                self.pop_size = min(self.pop_size + 1, 2 * self.initial_pop_size)
                self.population = np.vstack((self.population, np.random.uniform(-5, 5, size=(1, self.dim))))
                self.fitness = np.append(self.fitness, np.inf)
                self.velocities = np.vstack((self.velocities, np.zeros((1, self.dim))))
                self.best_positions = np.vstack((self.best_positions, self.population[-1].copy()))
                self.best_fitness = np.append(self.best_fitness, np.inf)
            elif improvement_rate < 0.05 and self.pop_size > self.initial_pop_size // 2:
                self.pop_size = max(self.pop_size - 1, self.initial_pop_size // 2)
                self.population = self.population[:self.pop_size]
                self.fitness = self.fitness[:self.pop_size]
                self.velocities = self.velocities[:self.pop_size]
                self.best_positions = self.best_positions[:self.pop_size]
                self.best_fitness = self.best_fitness[:self.pop_size]

        return self.global_best_fitness, self.global_best_position