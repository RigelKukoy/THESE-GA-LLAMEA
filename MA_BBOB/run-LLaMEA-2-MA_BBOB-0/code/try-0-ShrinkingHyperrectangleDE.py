import numpy as np

class ShrinkingHyperrectangleDE:
    def __init__(self, budget=10000, dim=10, pop_size=40, shrinkage_rate=0.99, F=0.7, CR=0.8, cauchy_scale=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.shrinkage_rate = shrinkage_rate
        self.F = F
        self.CR = CR
        self.cauchy_scale = cauchy_scale
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        self.population = np.random.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size

        best_index = np.argmin(fitness)
        self.f_opt = fitness[best_index]
        self.x_opt = self.population[best_index]
        
        current_lb = np.full(self.dim, self.lb)
        current_ub = np.full(self.dim, self.ub)


        while self.budget > 0:
            for i in range(self.pop_size):
                # Mutation (Cauchy)
                cauchy_noise = np.random.standard_cauchy(size=self.dim) * self.cauchy_scale
                mutant = self.population[i] + cauchy_noise
                mutant = np.clip(mutant, current_lb, current_ub)

                # Crossover
                crossover = np.random.uniform(size=self.dim) < self.CR
                trial = np.where(crossover, mutant, self.population[i])
                trial = np.clip(trial, current_lb, current_ub)

                # Selection
                f_trial = func(trial)
                self.budget -= 1

                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial

                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    self.population[i] = trial
                    
            # Shrink the hyperrectangle
            best_index = np.argmin(fitness)
            best_solution = self.population[best_index]
            
            range_width = (current_ub - current_lb) * self.shrinkage_rate
            current_lb = best_solution - range_width / 2
            current_ub = best_solution + range_width / 2
            
            current_lb = np.maximum(current_lb, np.full(self.dim, self.lb))
            current_ub = np.minimum(current_ub, np.full(self.dim, self.ub))

            # Repopulate the solutions within the shrinked hyperrectangle
            self.population = np.random.uniform(current_lb, current_ub, size=(self.pop_size, self.dim))
            fitness = np.array([func(x) for x in self.population])
            self.budget -= self.pop_size
            
            best_index_local = np.argmin(fitness)
            if fitness[best_index_local] < self.f_opt:
                 self.f_opt = fitness[best_index_local]
                 self.x_opt = self.population[best_index_local]
        

        return self.f_opt, self.x_opt