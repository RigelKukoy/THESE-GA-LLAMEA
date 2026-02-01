import numpy as np

class SelfAdjustingDE:
    def __init__(self, budget=10000, dim=10, popsize_init=None, F=0.5, CR=0.7, stagnation_threshold=10):
        self.budget = budget
        self.dim = dim
        self.popsize_init = popsize_init if popsize_init is not None else 10 * dim
        self.F = F
        self.CR = CR
        self.stagnation_threshold = stagnation_threshold
        self.population = None
        self.fitness = None
        self.f_opt = np.inf
        self.x_opt = None
        self.eval_count = 0
        self.stagnation_counter = 0
        self.best_fitness_history = []

    def initialize(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.popsize_init, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.eval_count += self.popsize_init
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]
        self.best_fitness_history.append(self.f_opt)

    def mutate(self, pop, F):
        mutated_pop = np.zeros_like(pop)
        for i in range(len(pop)):
            idxs = np.random.choice(len(pop), 3, replace=False)
            x_r1, x_r2, x_r3 = pop[idxs]
            mutated_pop[i] = x_r1 + F * (x_r2 - x_r3)
        return mutated_pop

    def crossover(self, pop, mutated_pop, CR):
        crossed_pop = np.zeros_like(pop)
        for i in range(len(pop)):
            for j in range(self.dim):
                if np.random.rand() < CR:
                    crossed_pop[i, j] = mutated_pop[i, j]
                else:
                    crossed_pop[i, j] = pop[i, j]
        return crossed_pop

    def mirrored_sampling(self, func, x):
        """Handles boundary constraints using mirrored sampling."""
        lb = func.bounds.lb
        ub = func.bounds.ub
        
        x_corrected = np.copy(x)
        for i in range(self.dim):
            if x[i] < lb:
                x_corrected[i] = lb + (lb - x[i])
            elif x[i] > ub:
                x_corrected[i] = ub - (x[i] - ub)
                
            # Double Mirroring to handle corner cases
            if x_corrected[i] < lb:
                x_corrected[i] = lb + (lb - x_corrected[i])
            elif x_corrected[i] > ub:
                x_corrected[i] = ub - (x_corrected[i] - ub)
                
        return x_corrected
        
    def __call__(self, func):
        self.initialize(func)
        popsize = self.popsize_init

        while self.eval_count < self.budget:
            # Adaptive F and CR
            F = np.random.normal(self.F, 0.1, popsize)
            F = np.clip(F, 0.1, 1.0)
            CR = np.random.normal(self.CR, 0.1, popsize)
            CR = np.clip(CR, 0.1, 1.0)

            mutated_population = self.mutate(self.population, F=np.mean(F))
            crossed_population = self.crossover(self.population, mutated_population, CR=np.mean(CR))
            
            # Boundary Handling with Mirrored Sampling
            for i in range(popsize):
                crossed_population[i] = self.mirrored_sampling(func, crossed_population[i])

            new_fitness = np.array([func(x) for x in crossed_population])
            self.eval_count += popsize

            # Selection
            for i in range(popsize):
                if new_fitness[i] < self.fitness[i]:
                    self.fitness[i] = new_fitness[i]
                    self.population[i] = crossed_population[i]

            # Update optimal solution
            current_best_fitness = np.min(self.fitness)
            if current_best_fitness < self.f_opt:
                self.f_opt = current_best_fitness
                self.x_opt = self.population[np.argmin(self.fitness)]
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1

            self.best_fitness_history.append(current_best_fitness)

            # Population size adjustment
            if self.stagnation_counter > self.stagnation_threshold:
                popsize = int(popsize * 0.9)  # Reduce population size
                if popsize < 4:
                    popsize = self.popsize_init # Reset to initial population size
                
                # Repopulate the population with better individuals
                best_indices = np.argsort(self.fitness)[:popsize]
                self.population = self.population[best_indices]
                self.fitness = self.fitness[best_indices]
                
                remaining_size = self.popsize_init - popsize
                new_individuals = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(remaining_size, self.dim))
                new_fitness_vals = np.array([func(x) for x in new_individuals])
                self.eval_count += remaining_size
                
                self.population = np.concatenate((self.population, new_individuals), axis=0)
                self.fitness = np.concatenate((self.fitness, new_fitness_vals), axis=0)
                popsize = self.popsize_init

                self.stagnation_counter = 0
            else:
                 if popsize < self.popsize_init:
                    popsize = self.popsize_init # Reset if population dropped earlier

            if self.eval_count >= self.budget:
                break

        return self.f_opt, self.x_opt