import numpy as np

class SelfAdaptiveDE:
    def __init__(self, budget=10000, dim=10, initial_popsize=None, F=0.5, CR=0.7, popsize_reduction_factor=0.9):
        self.budget = budget
        self.dim = dim
        self.initial_popsize = initial_popsize if initial_popsize is not None else 10 * self.dim
        self.popsize = self.initial_popsize
        self.F = F
        self.CR = CR
        self.popsize_reduction_factor = popsize_reduction_factor

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        self.population = np.random.uniform(lb, ub, size=(self.popsize, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.eval_count = self.popsize
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]
        
        generation = 0
        while self.eval_count < self.budget:
            generation += 1
            for i in range(self.popsize):
                # Simplified Mutation: Focus on exploitation using the best solution so far
                mutant = self.x_opt + self.F * (np.random.uniform(lb, ub, size=self.dim) - self.population[i]) #Simplified mutation with random vector

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
            
            #Adapt population size
            if generation % 10 == 0 and self.popsize > 5: 
                #Reduce population size gradually
                new_popsize = int(self.popsize * self.popsize_reduction_factor)
                
                if new_popsize < 5:
                    new_popsize = 5
                
                if new_popsize < self.popsize:
                    
                    sorted_indices = np.argsort(self.fitness)
                    
                    self.population = self.population[sorted_indices[:new_popsize]]
                    self.fitness = self.fitness[sorted_indices[:new_popsize]]
                    
                    self.popsize = new_popsize
                
            if self.eval_count > self.budget:
                break
        return self.f_opt, self.x_opt