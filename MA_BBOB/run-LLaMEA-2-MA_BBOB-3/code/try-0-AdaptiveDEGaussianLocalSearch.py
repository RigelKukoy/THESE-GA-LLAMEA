import numpy as np

class AdaptiveDEGaussianLocalSearch:
    def __init__(self, budget=10000, dim=10, pop_size=20, initial_mutation_factor=0.5, crossover_rate=0.7, local_search_frequency=10, local_search_sigma=0.1, mutation_adaptation_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.mutation_factor = initial_mutation_factor
        self.crossover_rate = crossover_rate
        self.local_search_frequency = local_search_frequency
        self.local_search_sigma = local_search_sigma
        self.mutation_adaptation_rate = mutation_adaptation_rate
        self.population = np.random.uniform(-5, 5, size=(pop_size, dim))
        self.fitness = np.zeros(pop_size)
        self.best_position = None
        self.best_fitness = np.inf
        self.eval_count = 0
        self.successful_mutations = 0
        self.mutation_success_rate = 0.5  # Initialize with a reasonable value

    def __call__(self, func):
        self.eval_count = 0
        self.successful_mutations = 0

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
                # Differential Evolution
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
                        self.population[i] = trial_vector
                        self.fitness[i] = f_trial
                        if f_trial < self.best_fitness:
                            self.best_fitness = f_trial
                            self.best_position = self.population[i].copy()
                        self.successful_mutations += 1

            # Local Search around best solution
            if self.eval_count // self.local_search_frequency == self.eval_count / self.local_search_frequency:
                for _ in range(5): # Perform a few local search steps
                    if self.eval_count < self.budget:
                        # Generate a Gaussian perturbation around the best solution
                        perturbed_solution = self.best_position + np.random.normal(0, self.local_search_sigma, self.dim)
                        perturbed_solution = np.clip(perturbed_solution, func.bounds.lb, func.bounds.ub)
                        f_perturbed = func(perturbed_solution)

                        self.eval_count += 1
                        if f_perturbed < self.best_fitness:
                            self.best_fitness = f_perturbed
                            self.best_position = perturbed_solution.copy()

            # Adapt mutation factor
            self.mutation_success_rate = self.successful_mutations / self.pop_size
            if self.mutation_success_rate > 0.2:
                self.mutation_factor *= (1 - self.mutation_adaptation_rate)  # Reduce mutation if successful
            else:
                self.mutation_factor /= (1 - self.mutation_adaptation_rate)  # Increase mutation if unsuccessful

            self.mutation_factor = np.clip(self.mutation_factor, 0.1, 1.0)  # Keep mutation within reasonable bounds
            self.successful_mutations = 0 # reset successful mutations

        return self.best_fitness, self.best_position