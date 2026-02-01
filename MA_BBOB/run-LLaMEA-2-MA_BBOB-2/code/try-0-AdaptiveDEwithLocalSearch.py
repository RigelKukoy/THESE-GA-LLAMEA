import numpy as np

class AdaptiveDEwithLocalSearch:
    def __init__(self, budget=10000, dim=10, popsize=None, F=0.5, CR=0.7, local_search_iterations=5, local_search_stepsize=0.1, stagnation_threshold=1e-6, stagnation_iterations=500):
        self.budget = budget
        self.dim = dim
        self.popsize = popsize if popsize is not None else 10 * self.dim
        self.F = F
        self.CR = CR
        self.local_search_iterations = local_search_iterations
        self.local_search_stepsize = local_search_stepsize
        self.stagnation_threshold = stagnation_threshold
        self.stagnation_iterations = stagnation_iterations
        self.best_fitness_history = []

    def local_search(self, func, x):
        x_current = x.copy()
        f_current = func(x_current)
        eval_count = 1
        for _ in range(self.local_search_iterations):
            step = np.random.uniform(-self.local_search_stepsize, self.local_search_stepsize, size=self.dim)
            x_new = np.clip(x_current + step, func.bounds.lb, func.bounds.ub)
            f_new = func(x_new)
            eval_count += 1
            if f_new < f_current:
                f_current = f_new
                x_current = x_new
        return f_current, x_current, eval_count

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        self.population = np.random.uniform(lb, ub, size=(self.popsize, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.eval_count = self.popsize
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]

        self.best_fitness_history.append(self.f_opt)
        stagnation_counter = 0

        while self.eval_count < self.budget:
            for i in range(self.popsize):
                # Mutation: Combination of current-to-best and rand/1
                best_idx = np.argmin(self.fitness)
                x_best = self.population[best_idx]
                idxs = np.random.choice(self.popsize, 3, replace=False)
                x1, x2, x3 = self.population[idxs]
                
                mutant = self.population[i] + self.F * (x_best - self.population[i]) + self.F * (x2 - x3)
                mutant = np.clip(mutant, lb, ub)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, self.population[i])

                # Selection
                f_trial = func(trial)
                self.eval_count += 1

                if f_trial < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = f_trial
                    if self.fitness[i] < self.f_opt:
                        self.f_opt = self.fitness[i]
                        self.x_opt = self.population[i]
            
            # Stagnation detection
            self.best_fitness_history.append(self.f_opt)
            if len(self.best_fitness_history) > self.stagnation_iterations:
                self.best_fitness_history.pop(0)
                if np.std(self.best_fitness_history) < self.stagnation_threshold:
                    stagnation_counter += 1
                    if stagnation_counter >= 2:
                        # Apply Local Search to the best individual
                        f_local, x_local, ls_eval_count = self.local_search(func, self.x_opt)
                        self.eval_count += ls_eval_count
                        if f_local < self.f_opt:
                            self.f_opt = f_local
                            self.x_opt = x_local
                        
                        # Restart the population around the best individual
                        self.population = np.random.normal(self.x_opt, (ub - lb) * 0.05, size=(self.popsize, self.dim))
                        self.population = np.clip(self.population, lb, ub)
                        self.fitness = np.array([func(x) for x in self.population])
                        self.eval_count += self.popsize
                        best_idx = np.argmin(self.fitness)
                        self.f_opt = self.fitness[best_idx]
                        self.x_opt = self.population[best_idx]
                        self.best_fitness_history = [self.f_opt] # Reset fitness history
                        stagnation_counter = 0  # Reset stagnation counter

        return self.f_opt, self.x_opt