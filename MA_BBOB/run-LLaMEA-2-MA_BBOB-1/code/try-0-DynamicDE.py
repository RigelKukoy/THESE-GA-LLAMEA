import numpy as np

class DynamicDE:
    def __init__(self, budget=10000, dim=10, initial_popsize=50, min_popsize=10, max_popsize=100, stagnation_threshold=500):
        self.budget = budget
        self.dim = dim
        self.initial_popsize = initial_popsize
        self.min_popsize = min_popsize
        self.max_popsize = max_popsize
        self.stagnation_threshold = stagnation_threshold
        self.popsize = initial_popsize
        self.population = None
        self.fitness = None
        self.best_fitness = np.inf
        self.best_solution = None
        self.F = 0.5
        self.CR = 0.7
        self.evals = 0
        self.stagnation_counter = 0

    def initialize_population(self, func):
        self.popsize = self.initial_popsize
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.popsize, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.best_fitness = np.min(self.fitness)
        self.best_solution = self.population[np.argmin(self.fitness)]
        self.evals = self.popsize

    def mutate(self, i):
        indices = np.random.choice(self.popsize, 3, replace=False)
        x_r1, x_r2, x_r3 = self.population[indices]
        v = x_r1 + self.F * (x_r2 - x_r3)
        return v

    def crossover(self, i, mutant):
        trial_vector = np.copy(self.population[i])
        j_rand = np.random.randint(self.dim)
        for j in range(self.dim):
            if np.random.rand() < self.CR or j == j_rand:
                trial_vector[j] = mutant[j]
        return trial_vector

    def handle_boundary(self, trial_vector, func):
        return np.clip(trial_vector, func.bounds.lb, func.bounds.ub)

    def adjust_population_size(self):
        if self.stagnation_counter > self.stagnation_threshold:
            # Increase population size to promote exploration
            self.popsize = min(self.popsize * 2, self.max_popsize)
            self.stagnation_counter = 0  # Reset stagnation counter
        elif self.popsize > self.initial_popsize and self.evals > self.budget * 0.75:
             # Reduce population size to promote exploitation towards the end
             self.popsize = max(self.popsize // 2, self.min_popsize)

    def restart_population(self, func):
        # Re-initialize the population if stagnation is detected
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.popsize, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        best_index = np.argmin(self.fitness)
        if self.fitness[best_index] < self.best_fitness:
            self.best_fitness = self.fitness[best_index]
            self.best_solution = self.population[best_index]
        self.evals += self.popsize

    def __call__(self, func):
        self.initialize_population(func)

        while self.evals < self.budget:
            best_fitness_before_gen = self.best_fitness  # Store best fitness before the generation
            for i in range(self.popsize):
                mutant = self.mutate(i)
                trial_vector = self.crossover(i, mutant)
                trial_vector = self.handle_boundary(trial_vector, func)

                f_trial = func(trial_vector)
                self.evals += 1

                if f_trial < self.fitness[i]:
                    self.population[i] = trial_vector
                    self.fitness[i] = f_trial

                    if f_trial < self.best_fitness:
                        self.best_fitness = f_trial
                        self.best_solution = trial_vector

                if self.evals >= self.budget:
                    break
            
            if self.best_fitness >= best_fitness_before_gen:
                self.stagnation_counter += self.popsize  # Increment stagnation counter

            if self.stagnation_counter > self.stagnation_threshold and self.evals < self.budget * 0.8:
                self.restart_population(func)
                self.stagnation_counter = 0

            self.adjust_population_size() # dynamic population sizing
           

            # Adaptive F and CR (simplified)
            self.F = np.random.uniform(0.4, 0.9)
            self.CR = np.random.uniform(0.3, 1.0)

        return self.best_fitness, self.best_solution