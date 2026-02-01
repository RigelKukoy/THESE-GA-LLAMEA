import numpy as np

class AdaptiveHybridDEPSOGaussian:
    def __init__(self, budget=10000, dim=10, initial_pop_size=20, de_mutation_factor=0.5, pso_inertia=0.7, pso_cognitive=1.5, pso_social=1.5, gaussian_mutation_rate=0.1, diversity_threshold=0.1):
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
        self.gaussian_mutation_rate = gaussian_mutation_rate
        self.diversity_threshold = diversity_threshold
        self.eval_count = 0
        self.stagnation_counter = 0
        self.max_stagnation = 100
        self.previous_best_fitness = np.inf
        self.adaptive_mutation_factor = de_mutation_factor # Adaptive DE mutation
        self.adaptive_inertia = pso_inertia #Adaptive PSO inertia

    def calculate_diversity(self):
        """Calculates population diversity based on the mean pairwise distance."""
        distances = np.linalg.norm(self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :], axis=2)
        diversity = np.mean(distances)
        return diversity
    
    def gaussian_mutation(self, x):
        """Applies Gaussian mutation to a given solution."""
        mutation = np.random.normal(0, self.gaussian_mutation_rate, size=self.dim)
        return x + mutation

    def restart_population(self):
         """Restarts the population with new random solutions."""
         self.population = np.random.uniform(-5, 5, size=(self.pop_size, self.dim))
         self.fitness = np.zeros(self.pop_size)
         self.velocities = np.zeros((self.pop_size, self.dim))
         self.best_positions = self.population.copy()
         self.best_fitness = np.full(self.pop_size, np.inf)

         # Re-evaluate the restarted population
         for i in range(self.pop_size):
             if self.eval_count < self.budget:
                 f = func(self.population[i])
                 self.eval_count += 1
                 self.fitness[i] = f
                 if f < self.best_fitness[i]:
                     self.best_fitness[i] = f
                     self.best_positions[i] = self.population[i].copy()
                     if f < self.global_best_fitness:
                         self.global_best_fitness = f
                         self.global_best_position = self.population[i].copy()


    def __call__(self, func):
        self.eval_count = 0

        # Initial evaluation
        for i in range(self.pop_size):
            if self.eval_count < self.budget:
                f = func(self.population[i])
                self.eval_count += 1
                self.fitness[i] = f
                if f < self.best_fitness[i]:
                    self.best_fitness[i] = f
                    self.best_positions[i] = self.population[i].copy()
                    if f < self.global_best_fitness:
                        self.global_best_fitness = f
                        self.global_best_position = self.population[i].copy()
        self.previous_best_fitness = self.global_best_fitness
        while self.eval_count < self.budget:
            for i in range(self.pop_size):
                # Choose between DE, PSO, and Gaussian mutation
                rand = np.random.rand()
                if rand < 0.33:
                    # Differential Evolution
                    r1, r2, r3 = np.random.choice(self.pop_size, 3, replace=False)
                    new_position = self.best_positions[r1] + self.adaptive_mutation_factor * (self.population[r2] - self.population[r3])
                    new_position = np.clip(new_position, func.bounds.lb, func.bounds.ub)
                elif rand < 0.66:
                    # Particle Swarm Optimization
                    new_velocity = self.adaptive_inertia * self.velocities[i] + \
                                   self.pso_cognitive * np.random.rand(self.dim) * (self.best_positions[i] - self.population[i]) + \
                                   self.pso_social * np.random.rand(self.dim) * (self.global_best_position - self.population[i])
                    new_position = self.population[i] + new_velocity
                    new_position = np.clip(new_position, func.bounds.lb, func.bounds.ub)
                    self.velocities[i] = new_velocity
                else:
                    # Gaussian Mutation
                    new_position = self.gaussian_mutation(self.population[i])
                    new_position = np.clip(new_position, func.bounds.lb, func.bounds.ub)

                # Evaluate new position
                f = func(new_position)
                self.eval_count += 1
                if self.eval_count >= self.budget:
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
            #Adapt mutation factor
            self.adaptive_mutation_factor = 0.5 + 0.5 * np.exp(-10 * self.eval_count / self.budget)
            self.adaptive_inertia = 0.9 - 0.7 * (self.eval_count / self.budget)

            # Stagnation check and restart
            if self.global_best_fitness >= self.previous_best_fitness:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0

            if self.stagnation_counter > self.max_stagnation:
                diversity = self.calculate_diversity()
                if diversity < self.diversity_threshold:
                    self.restart_population()
                    self.stagnation_counter = 0

            self.previous_best_fitness = self.global_best_fitness

        return self.global_best_fitness, self.global_best_position