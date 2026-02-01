import numpy as np

class AdaptiveDEAgingRestart:
    def __init__(self, budget=10000, dim=10, initial_pop_size=50, F=0.5, CR=0.9, local_search_prob=0.1, stagnation_threshold=100, restart_frequency=500):
        self.budget = budget
        self.dim = dim
        self.pop_size = initial_pop_size
        self.initial_pop_size = initial_pop_size
        self.F = F
        self.CR = CR
        self.local_search_prob = local_search_prob
        self.stagnation_threshold = stagnation_threshold
        self.restart_frequency = restart_frequency
        self.ages = np.zeros(self.pop_size)  # Initialize ages for each individual

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
        stagnation_counter = 0

        while self.budget > 0:
            generation += 1
            new_population = np.copy(self.population)
            new_fitness = np.copy(fitness)
            
            # Increase ages of all individuals
            self.ages += 1

            for i in range(self.pop_size):
                # Differential Evolution
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                x_r1, x_r2, x_r3 = self.population[idxs]
                mutant = x_r1 + self.F * (x_r2 - x_r3)
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)
                
                crossover = np.random.uniform(size=self.dim) < self.CR
                trial = np.where(crossover, mutant, self.population[i])
                
                # Local Search with Probability
                if np.random.rand() < self.local_search_prob:
                    trial = trial + np.random.normal(0, 0.05, size=self.dim)
                    trial = np.clip(trial, func.bounds.lb, func.bounds.ub)
                
                # Selection
                f_trial = func(trial)
                self.budget -= 1
                
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial
                    stagnation_counter = 0  # Reset stagnation counter

                if f_trial < fitness[i]:
                    new_fitness[i] = f_trial
                    new_population[i] = trial
                    self.ages[i] = 0 # Reset age if individual improves
                else:
                    new_fitness[i] = fitness[i]
                    new_population[i] = self.population[i]
                    
            self.population = new_population
            fitness = new_fitness

            # Adaptive F and CR
            self.F = np.clip(np.random.normal(0.5, 0.1), 0.1, 0.9)
            self.CR = np.clip(np.random.normal(0.9, 0.1), 0.1, 0.9)

            # Stagnation Check and Restart
            stagnation_counter += 1
            if stagnation_counter > self.stagnation_threshold or generation % self.restart_frequency == 0:
                # Introduce diversity by re-initializing a portion of the population
                num_to_reinitialize = int(0.3 * self.pop_size)
                idxs_to_reinitialize = np.argsort(self.ages)[-num_to_reinitialize:]  # Reinitialize oldest individuals
                self.population[idxs_to_reinitialize] = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(num_to_reinitialize, self.dim))
                fitness[idxs_to_reinitialize] = np.array([func(x) for x in self.population[idxs_to_reinitialize]])
                self.budget -= num_to_reinitialize
                self.ages[idxs_to_reinitialize] = 0 # Reset age of reinitialized individuals
                stagnation_counter = 0  # Reset stagnation counter

                for i in idxs_to_reinitialize:
                    if fitness[i] < self.f_opt:
                        self.f_opt = fitness[i]
                        self.x_opt = self.population[i]


        return self.f_opt, self.x_opt