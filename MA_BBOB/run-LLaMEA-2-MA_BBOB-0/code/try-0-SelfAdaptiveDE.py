import numpy as np

class SelfAdaptiveDE:
    def __init__(self, budget=10000, dim=10, initial_pop_size=50, min_pop_size=10, max_pop_size=100, initial_F=0.5, initial_CR=0.9, age_limit=50):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = initial_pop_size
        self.min_pop_size = min_pop_size
        self.max_pop_size = max_pop_size
        self.pop_size = initial_pop_size
        self.F = initial_F
        self.CR = initial_CR
        self.age_limit = age_limit
        self.population = None
        self.fitness = None
        self.ages = None
        self.f_opt = np.Inf
        self.x_opt = None

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.ages = np.zeros(self.pop_size, dtype=int)
        self.budget -= self.pop_size

        for i in range(self.pop_size):
            if self.fitness[i] < self.f_opt:
                self.f_opt = self.fitness[i]
                self.x_opt = self.population[i]

    def adjust_population_size(self):
        # Dynamically adjust population size based on stagnation
        if np.std(self.fitness) < 1e-6 and self.pop_size < self.max_pop_size:  #Stagnation detected.
            self.pop_size = min(self.pop_size + 5, self.max_pop_size) #Increase the population size.
        elif self.pop_size > self.initial_pop_size and np.random.rand() < 0.05: # Reduce population if conditions are favorable
            self.pop_size = max(self.pop_size - 2, self.min_pop_size)

    def age_population(self):
        # Increment age for each individual
        self.ages += 1

        # Identify and remove old individuals (replace with new ones)
        old_indices = np.where(self.ages >= self.age_limit)[0]
        num_old = len(old_indices)

        if num_old > 0:
            self.population[old_indices] = np.random.uniform(self.population.min(), self.population.max(), size=(num_old, self.dim)) #func.bounds.lb, func.bounds.ub, size=(num_old, self.dim)
            self.fitness[old_indices] = np.array([np.inf] * num_old) # To force re-evaluation
            self.ages[old_indices] = 0 #Reset age.

    def __call__(self, func):
        self.initialize_population(func)
        generation = 0
        while self.budget > 0:
            generation += 1

            self.adjust_population_size()
            if self.pop_size != len(self.population): # population size was changed.
                self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
                self.fitness = np.array([func(x) for x in self.population])
                self.ages = np.zeros(self.pop_size, dtype=int)
                self.budget -= self.pop_size
                for i in range(self.pop_size):
                    if self.fitness[i] < self.f_opt:
                        self.f_opt = self.fitness[i]
                        self.x_opt = self.population[i]

            new_population = np.copy(self.population)
            new_fitness = np.copy(self.fitness)

            for i in range(self.pop_size):
                # Differential Evolution
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                x_r1, x_r2, x_r3 = self.population[idxs]
                mutant = x_r1 + self.F * (x_r2 - x_r3)
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)
                
                crossover = np.random.uniform(size=self.dim) < self.CR
                trial = np.where(crossover, mutant, self.population[i])
                
                # Selection
                f_trial = func(trial)
                self.budget -= 1
                
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial
                    
                if f_trial < self.fitness[i]:
                    new_fitness[i] = f_trial
                    new_population[i] = trial
                    self.ages[i] = 0 # Reset age if improved.
                else:
                    new_fitness[i] = self.fitness[i]
                    new_population[i] = self.population[i]

            self.population = new_population
            self.fitness = new_fitness
            self.age_population()

        return self.f_opt, self.x_opt