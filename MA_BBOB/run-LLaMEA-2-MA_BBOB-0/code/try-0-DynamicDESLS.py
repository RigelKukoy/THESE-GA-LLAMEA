import numpy as np

class DynamicDESLS:
    def __init__(self, budget=10000, dim=10, initial_pop_size=50, stagnation_threshold=0.001, ls_probability=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = initial_pop_size
        self.initial_pop_size = initial_pop_size
        self.stagnation_threshold = stagnation_threshold
        self.ls_probability = ls_probability
        self.F = 0.5
        self.CR = 0.9
        self.best_fitness_history = []
        self.shrink_factor = 0.9
        self.grow_factor = 1.1

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
        
        self.best_fitness_history.append(self.f_opt)

        while self.budget > 0:
            for i in range(self.pop_size):
                # Mutation
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                x_r1, x_r2, x_r3 = self.population[idxs]
                mutant = x_r1 + self.F * (x_r2 - x_r3)
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                # Crossover
                crossover = np.random.uniform(size=self.dim) < self.CR
                trial = np.where(crossover, mutant, self.population[i])

                # Selection
                f_trial = func(trial)
                self.budget -= 1
                
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial

                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    self.population[i] = trial
                
                # Local Search
                if np.random.rand() < self.ls_probability and self.budget > 0:
                    x_ls = self.local_search(func, trial, func.bounds.lb, func.bounds.ub)
                    f_ls = func(x_ls)
                    self.budget -= 1
                    if f_ls < self.f_opt:
                        self.f_opt = f_ls
                        self.x_opt = x_ls
                    if f_ls < fitness[i]:
                        fitness[i] = f_ls
                        self.population[i] = x_ls

            # Dynamic Population Sizing
            if len(self.best_fitness_history) > 10:
                if np.std(self.best_fitness_history[-10:]) < self.stagnation_threshold:
                    # Stagnation detected, shrink population
                    self.pop_size = max(int(self.pop_size * self.shrink_factor), 10)  # Ensure minimum size
                    self.population = self.population[:self.pop_size]
                    fitness = fitness[:self.pop_size]
                    #Increase local search probability when stagnated
                    self.ls_probability = min(self.ls_probability * 1.2, 0.5)
                else:
                    # Increase Population Size
                     self.pop_size = min(int(self.pop_size * self.grow_factor), self.initial_pop_size * 2)
                     if self.pop_size > len(self.population):
                        num_to_add = self.pop_size - len(self.population)
                        new_individuals = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(num_to_add, self.dim))
                        new_fitness = np.array([func(x) for x in new_individuals])
                        self.budget -= num_to_add
                        self.population = np.vstack((self.population, new_individuals))
                        fitness = np.concatenate((fitness, new_fitness))
                        for i in range(len(fitness)-num_to_add, len(fitness)):
                            if fitness[i] < self.f_opt:
                                self.f_opt = fitness[i]
                                self.x_opt = self.population[i]
                     #Decrease local search probability when not stagnated
                     self.ls_probability = max(self.ls_probability / 1.2, 0.01)

            self.best_fitness_history.append(self.f_opt)
            if self.budget <= 0:
                break

        return self.f_opt, self.x_opt

    def local_search(self, func, x, lb, ub, radius=0.1, iterations=5):
        x_best = x.copy()
        f_best = func(x_best)
        for _ in range(iterations):
            x_new = x_best + np.random.uniform(-radius, radius, size=self.dim)
            x_new = np.clip(x_new, lb, ub)
            f_new = func(x_new)
            if f_new < f_best:
                f_best = f_new
                x_best = x_new
        return x_best