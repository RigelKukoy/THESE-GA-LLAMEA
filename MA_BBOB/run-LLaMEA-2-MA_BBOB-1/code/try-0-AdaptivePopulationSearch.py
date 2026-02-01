import numpy as np

class AdaptivePopulationSearch:
    def __init__(self, budget=10000, dim=10, pop_size=50, de_rate=0.7, ls_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.de_rate = de_rate
        self.ls_rate = ls_rate

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size
        
        best_index = np.argmin(fitness)
        self.f_opt = fitness[best_index]
        self.x_opt = population[best_index]
        
        archive = []

        while self.budget > 0:
            # Adaptive strategy selection
            if np.random.rand() < self.de_rate:
                # Differential Evolution
                for i in range(self.pop_size):
                    if self.budget <= 0:
                        break

                    idxs = np.random.choice(self.pop_size, 3, replace=False)
                    x_r1, x_r2, x_r3 = population[idxs]
                    
                    mutated = population[i] + 0.5 * (x_r2 - x_r3)
                    
                    # Crossover
                    j_rand = np.random.randint(self.dim)
                    for j in range(self.dim):
                        if np.random.rand() > 0.9 and j != j_rand:
                            mutated[j] = population[i][j]
                    
                    mutated = np.clip(mutated, func.bounds.lb, func.bounds.ub)    
                    f_mutated = func(mutated)
                    self.budget -= 1

                    if f_mutated < fitness[i]:
                        fitness[i] = f_mutated
                        population[i] = mutated

                        if f_mutated < self.f_opt:
                            self.f_opt = f_mutated
                            self.x_opt = mutated
                            
            elif np.random.rand() < self.ls_rate:
                # Local Search (perturb the best solution)
                if self.budget <= 0:
                    break
                
                x_perturbed = self.x_opt + np.random.normal(0, 0.1, size=self.dim)
                x_perturbed = np.clip(x_perturbed, func.bounds.lb, func.bounds.ub)
                
                f_perturbed = func(x_perturbed)
                self.budget -= 1
                
                if f_perturbed < self.f_opt:
                    self.f_opt = f_perturbed
                    self.x_opt = x_perturbed
            else:
                # Global search, random restart with probability 1-de_rate-ls_rate
                if self.budget <= 0:
                    break
                
                x_rand = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
                f_rand = func(x_rand)
                self.budget -= 1
                
                if f_rand < self.f_opt:
                    self.f_opt = f_rand
                    self.x_opt = x_rand

            best_index = np.argmin(fitness)
            if fitness[best_index] < self.f_opt:
              self.f_opt = fitness[best_index]
              self.x_opt = population[best_index]
            
        return self.f_opt, self.x_opt