import numpy as np

class SelfOrganizingDE:
    def __init__(self, budget=10000, dim=10, pop_size=20, initial_mutation_factor=0.5, crossover_rate=0.7, learning_rate=0.1, diversity_threshold=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.mutation_factor = initial_mutation_factor
        self.crossover_rate = crossover_rate
        self.learning_rate = learning_rate
        self.diversity_threshold = diversity_threshold
        self.population = np.random.uniform(-5, 5, size=(pop_size, dim))
        self.fitness = np.zeros(pop_size)
        self.best_position = None
        self.best_fitness = np.inf
        self.eval_count = 0
        self.success_history = []

    def calculate_diversity(self):
        """Calculates the diversity of the population based on the standard deviation of each dimension."""
        std_devs = np.std(self.population, axis=0)
        return np.mean(std_devs)

    def adjust_mutation_factor(self, success_rate):
        """Adjusts the mutation factor based on the recent success rate of generating better solutions."""
        if self.success_history:
            success_rate = np.mean(self.success_history[-10:])  # Consider the last 10 updates
            self.mutation_factor += self.learning_rate * (success_rate - 0.5)  # Adjust towards 0.5
            self.mutation_factor = np.clip(self.mutation_factor, 0.1, 1.0) # Keep mutation factor bounded
        return self.mutation_factor

    def __call__(self, func):
        self.eval_count = 0
        self.success_history = []

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
            diversity = self.calculate_diversity()
            
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
                    if f_trial < self.fitness[i]:
                        self.success_history.append(1)  # Record a success
                        self.population[i] = trial_vector
                        self.fitness[i] = f_trial
                        if f_trial < self.best_fitness:
                            self.best_fitness = f_trial
                            self.best_position = self.population[i].copy()
                    else:
                        self.success_history.append(0)  # Record a failure

            # Dynamically Adjust Mutation Factor
            self.mutation_factor = self.adjust_mutation_factor(np.mean(self.success_history[-10:] if len(self.success_history) > 0 else [0]))
            
            # If diversity is too low, increase mutation factor slightly
            if diversity < self.diversity_threshold:
                self.mutation_factor = min(self.mutation_factor + 0.1, 1.0)

        return self.best_fitness, self.best_position