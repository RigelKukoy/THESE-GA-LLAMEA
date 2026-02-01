import numpy as np

class AdaptiveLevyDE:
    def __init__(self, budget=10000, dim=10, pop_size=20, initial_mutation_factor=0.5, crossover_rate=0.7, levy_exponent=1.5):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.mutation_factor = initial_mutation_factor
        self.crossover_rate = crossover_rate
        self.levy_exponent = levy_exponent
        self.population = np.random.uniform(-5, 5, size=(pop_size, dim))
        self.fitness = np.zeros(pop_size)
        self.best_position = None
        self.best_fitness = np.inf
        self.eval_count = 0
        self.mutation_factor_memory = np.full(pop_size, initial_mutation_factor)

    def truncated_levy_flight(self, size, truncation_level=3):
        """
        Generate truncated Lévy flight steps.
        """
        u = np.random.randn(size)
        v = np.random.randn(size)
        step = u / abs(v)**(1/self.levy_exponent)
        step[step > truncation_level] = truncation_level
        step[step < -truncation_level] = -truncation_level
        return step

    def __call__(self, func):
        self.eval_count = 0

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
                # Self-Adaptive Mutation Factor
                if np.random.rand() < 0.1:  # Probability to adjust mutation factor
                    self.mutation_factor_memory[i] = np.random.uniform(0.1, 1.0)
                mutation_factor = self.mutation_factor_memory[i]

                # Differential Evolution with Truncated Lévy Flight Mutation
                r1, r2, r3 = np.random.choice(self.pop_size, 3, replace=False)
                levy_steps = self.truncated_levy_flight(self.dim)
                mutant_vector = self.population[r1] + mutation_factor * (self.population[r2] - self.population[r3]) + levy_steps
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

                    #Greedy selection: Bias towards fitter individuals
                    if f_trial < self.fitness[i]:
                        self.population[i] = trial_vector
                        self.fitness[i] = f_trial
                        if f_trial < self.best_fitness:
                            self.best_fitness = f_trial
                            self.best_position = self.population[i].copy()
                    else:
                        #If trial is worse, randomly replace with best to help convergence
                        if np.random.rand() < 0.05:
                            self.population[i] = self.best_position.copy()

        return self.best_fitness, self.best_position