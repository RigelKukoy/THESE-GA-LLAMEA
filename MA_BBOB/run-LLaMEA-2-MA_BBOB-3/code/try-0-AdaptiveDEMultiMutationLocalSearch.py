import numpy as np

class AdaptiveDEMultiMutationLocalSearch:
    def __init__(self, budget=10000, dim=10, pop_size=20, mutation_factor=0.5, crossover_rate=0.7, local_search_prob=0.1, initial_learning_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.mutation_factor = mutation_factor
        self.crossover_rate = crossover_rate
        self.local_search_prob = local_search_prob
        self.learning_rate = initial_learning_rate
        self.initial_learning_rate = initial_learning_rate
        self.population = np.random.uniform(-5, 5, size=(pop_size, dim))
        self.fitness = np.zeros(pop_size)
        self.best_position = None
        self.best_fitness = np.inf
        self.eval_count = 0

    def calculate_diversity(self):
        """Calculates the population diversity based on the variance of each dimension."""
        mean_position = np.mean(self.population, axis=0)
        diversity = np.mean(np.var(self.population, axis=0))
        return diversity

    def __call__(self, func):
        self.eval_count = 0
        self.learning_rate = self.initial_learning_rate # Reset learning rate

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
            # Adjust learning rate based on diversity
            self.learning_rate = self.initial_learning_rate * (diversity / 25) # Scale diversity to [0, 1] assuming bounds are [-5,5] and var max is 25

            for i in range(self.pop_size):
                # Combined Mutation Strategy
                if np.random.rand() < 0.5:
                    # Current-to-rand mutation
                    r1, r2 = np.random.choice(self.pop_size, 2, replace=False)
                    mutant_vector = self.population[i] + self.mutation_factor * (self.population[r1] - self.population[r2])
                else:
                    # Current-to-best mutation
                    r1 = np.random.choice(self.pop_size, 1, replace=False)[0]
                    mutant_vector = self.population[i] + self.mutation_factor * (self.best_position - self.population[r1])

                mutant_vector = np.clip(mutant_vector, func.bounds.lb, func.bounds.ub)

                # Crossover
                trial_vector = np.zeros(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.crossover_rate or j == np.random.randint(self.dim):
                        trial_vector[j] = mutant_vector[j]
                    else:
                        trial_vector[j] = self.population[i][j]

                # Adaptive Local Search around the best solution
                if np.random.rand() < self.local_search_prob:
                    perturbation = np.random.uniform(-self.learning_rate, self.learning_rate, self.dim)
                    trial_vector = self.best_position + perturbation
                    trial_vector = np.clip(trial_vector, func.bounds.lb, func.bounds.ub)

                # Evaluate trial vector
                f_trial = func(trial_vector) if self.eval_count < self.budget else np.inf
                if self.eval_count < self.budget:
                    self.eval_count += 1
                    if f_trial < self.fitness[i]:
                        self.population[i] = trial_vector
                        self.fitness[i] = f_trial
                        if f_trial < self.best_fitness:
                            self.best_fitness = f_trial
                            self.best_position = self.population[i].copy()

        return self.best_fitness, self.best_position