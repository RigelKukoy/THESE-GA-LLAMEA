import numpy as np

class HybridDEwithSA:
    def __init__(self, budget=10000, dim=10, pop_size=20, mutation_factor=0.5, crossover_rate=0.7, initial_temp=100, cooling_rate=0.95):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.mutation_factor = mutation_factor
        self.crossover_rate = crossover_rate
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.population = np.random.uniform(-5, 5, size=(pop_size, dim))
        self.fitness = np.zeros(pop_size)
        self.best_position = None
        self.best_fitness = np.inf
        self.eval_count = 0
        self.temperature = initial_temp

    def __call__(self, func):
        self.eval_count = 0
        self.temperature = self.initial_temp

        # Initialize fitness values
        for i in range(self.pop_size):
            if self.eval_count < self.budget:
                f = func(self.population[i])
                self.eval_count += 1
                self.fitness[i] = f
                if f < self.best_fitness:
                    self.best_fitness = f
                    self.best_position = self.population[i].copy()

        while self.eval_count < self.budget:
            for i in range(self.pop_size):
                # Differential Evolution Mutation
                r1, r2, r3 = np.random.choice(self.pop_size, 3, replace=False)
                mutant_vector = self.population[r1] + self.mutation_factor * (self.population[r2] - self.population[r3])
                mutant_vector = np.clip(mutant_vector, func.bounds.lb, func.bounds.ub)

                # Crossover
                trial_vector = np.zeros(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.crossover_rate or j == np.random.randint(self.dim):
                        trial_vector[j] = mutant_vector[j]
                    else:
                        trial_vector[j] = self.population[i][j]

                # Evaluate trial vector
                f_trial = func(trial_vector) if self.eval_count < self.budget else np.inf
                if self.eval_count < self.budget:
                    self.eval_count += 1

                    # Simulated Annealing acceptance criterion
                    delta_e = f_trial - self.fitness[i]
                    if delta_e < 0:
                        self.population[i] = trial_vector
                        self.fitness[i] = f_trial
                        if f_trial < self.best_fitness:
                            self.best_fitness = f_trial
                            self.best_position = self.population[i].copy()
                    else:
                        # Acceptance probability based on temperature
                        acceptance_prob = np.exp(-delta_e / self.temperature)
                        if np.random.rand() < acceptance_prob:
                            self.population[i] = trial_vector
                            self.fitness[i] = f_trial

            #Cooling
            self.temperature *= self.cooling_rate

        return self.best_fitness, self.best_position