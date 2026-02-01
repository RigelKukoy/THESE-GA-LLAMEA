import numpy as np

class AdaptiveLevyDEAging:
    def __init__(self, budget=10000, dim=10, pop_size=20, mutation_factor=0.5, crossover_rate=0.7, levy_exponent=1.5, aging_rate=0.01):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.mutation_factor = mutation_factor
        self.crossover_rate = crossover_rate
        self.levy_exponent = levy_exponent
        self.aging_rate = aging_rate
        self.population = np.random.uniform(-5, 5, size=(pop_size, dim))
        self.fitness = np.zeros(pop_size)
        self.age = np.zeros(pop_size)  # Initialize age for each individual
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

        # Initialize fitness values and best solution
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
                levy_steps = self.levy_flight(self.dim) * (1 + self.age[i] * self.aging_rate)  # Adapt step size based on age
                mutant_vector = self.population[r1] + self.mutation_factor * (self.population[r2] - self.population[r3]) + levy_steps
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
                        self.age[i] = 0 # reset age
                        if f_trial < self.best_fitness:
                            self.best_fitness = f_trial
                            self.best_position = self.population[i].copy()
                    else:
                        self.age[i] += 1  # Increment age if not improved
            
            # Increase age for all individuals
            #self.age += 1
            
            # Rejuvenate old individuals
            for i in range(self.pop_size):
                if self.age[i] > 100: # aging threshold
                    self.population[i] = np.random.uniform(-5, 5, self.dim)
                    self.fitness[i] = func(self.population[i]) if self.eval_count < self.budget else np.inf
                    if self.eval_count < self.budget:
                         self.eval_count += 1

                    self.age[i] = 0
                    if self.fitness[i] < self.best_fitness:
                        self.best_fitness = self.fitness[i]
                        self.best_position = self.population[i].copy()

        return self.best_fitness, self.best_position