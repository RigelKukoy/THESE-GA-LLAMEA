import numpy as np

class LevyDE:
    def __init__(self, budget=10000, dim=10, pop_size=20, mutation_factor=0.5, crossover_rate=0.7, levy_exponent=1.5, local_search_prob=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.mutation_factor = mutation_factor
        self.crossover_rate = crossover_rate
        self.levy_exponent = levy_exponent
        self.local_search_prob = local_search_prob
        self.population = np.random.uniform(-5, 5, size=(pop_size, dim))
        self.fitness = np.zeros(pop_size)
        self.best_position = None
        self.best_fitness = np.inf
        self.eval_count = 0

    def levy_flight(self, size):
        """
        Generate Lévy flight steps.
        """
        num = np.random.randn(size)
        den = np.abs(np.random.randn(size))**(1/self.levy_exponent)
        sigma = (np.math.gamma(1+self.levy_exponent) * np.sin(np.pi*self.levy_exponent/2) / (np.math.gamma((1+self.levy_exponent)/2) * self.levy_exponent * 2**((self.levy_exponent-1)/2)))**(1/self.levy_exponent)
        step = sigma * num / den
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
                # Differential Evolution with Lévy Flight Mutation
                r1, r2, r3 = np.random.choice(self.pop_size, 3, replace=False)
                levy_steps = self.levy_flight(self.dim)
                mutant_vector = self.population[r1] + self.mutation_factor * (self.population[r2] - self.population[r3]) + levy_steps
                mutant_vector = np.clip(mutant_vector, func.bounds.lb, func.bounds.ub)

                # Crossover
                trial_vector = np.zeros(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.crossover_rate or j == np.random.randint(self.dim):
                        trial_vector[j] = mutant_vector[j]
                    else:
                        trial_vector[j] = self.population[i][j]

                # Probabilistic Local Search around the best solution
                if np.random.rand() < self.local_search_prob:
                    trial_vector = self.best_position + np.random.uniform(-0.1, 0.1, self.dim) # Small perturbation
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