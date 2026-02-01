import numpy as np

class AdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, initial_F=0.5, initial_CR=0.9, diversity_threshold=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = initial_F
        self.CR = initial_CR
        self.initial_F = initial_F
        self.initial_CR = initial_CR
        self.diversity_threshold = diversity_threshold
        self.archive_size = int(pop_size * 0.2)  # Archive size is 20% of population size
        self.archive = []

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        
        for i in range(self.pop_size):
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = self.population[i]

        generation = 0
        while self.budget > 0:
            generation += 1
            new_population = np.copy(self.population)
            new_fitness = np.copy(fitness)

            # Calculate population diversity
            diversity = np.std(self.population)

            # Adjust F and CR based on diversity and improvement rate
            if diversity < self.diversity_threshold:
                self.F = self.initial_F + 0.2 * np.random.randn()  # Increase F for exploration
                self.CR = self.initial_CR - 0.1 * np.random.rand() # Decrease CR for exploration
            else:
                self.F = self.initial_F - 0.1 * np.random.randn()  # Decrease F for exploitation
                self.CR = self.initial_CR + 0.2 * np.random.rand() # Increase CR for exploitation

            self.F = np.clip(self.F, 0.1, 0.9)
            self.CR = np.clip(self.CR, 0.1, 0.9)

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
                    
                if f_trial < fitness[i]:
                    new_fitness[i] = f_trial
                    new_population[i] = trial

                    # Update archive with successful trials
                    if len(self.archive) < self.archive_size:
                        self.archive.append(trial)
                    else:
                        # Replace a random element in the archive
                        idx_to_replace = np.random.randint(0, self.archive_size)
                        self.archive[idx_to_replace] = trial
                else:
                    new_fitness[i] = fitness[i]
                    new_population[i] = self.population[i]

            self.population = new_population
            fitness = new_fitness

        return self.f_opt, self.x_opt