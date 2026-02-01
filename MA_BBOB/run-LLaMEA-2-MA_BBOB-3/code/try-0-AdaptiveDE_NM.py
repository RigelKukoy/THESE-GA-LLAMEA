import numpy as np
from scipy.optimize import minimize

class AdaptiveDE_NM:
    def __init__(self, budget=10000, dim=10, pop_size=20, de_mutation_factor=0.5, nm_max_iter=50):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.de_mutation_factor = de_mutation_factor
        self.nm_max_iter = nm_max_iter
        self.population = np.random.uniform(-5, 5, size=(self.pop_size, dim))
        self.fitness = np.zeros(self.pop_size)
        self.best_fitness = np.inf
        self.best_position = None
        self.stagnation_counter = 0
        self.stagnation_threshold = 50 # Number of iterations without improvement before switching to local search
        self.diversity_threshold = 0.1 # Threshold for population diversity to switch search strategy

    def __call__(self, func):
        eval_count = 0
        
        # Evaluate initial population
        for i in range(self.pop_size):
            if eval_count < self.budget:
                self.fitness[i] = func(self.population[i])
                eval_count += 1
                if self.fitness[i] < self.best_fitness:
                    self.best_fitness = self.fitness[i]
                    self.best_position = self.population[i].copy()

        while eval_count < self.budget:
            # Calculate population diversity
            diversity = np.std(self.population)

            if diversity > self.diversity_threshold:
                # Global Search with Differential Evolution
                new_population = np.zeros_like(self.population)
                for i in range(self.pop_size):
                    if eval_count >= self.budget:
                        break
                    r1, r2, r3 = np.random.choice(self.pop_size, 3, replace=False)
                    mutant_vector = self.population[r1] + self.de_mutation_factor * (self.population[r2] - self.population[r3])
                    
                    # Crossover (Binomial/Uniform)
                    crossover_mask = np.random.rand(self.dim) < 0.9
                    new_population[i] = np.where(crossover_mask, mutant_vector, self.population[i])
                    new_population[i] = np.clip(new_population[i], func.bounds.lb, func.bounds.ub)
                    
                    new_fitness = func(new_population[i])
                    eval_count += 1
                    
                    if new_fitness < self.fitness[i]:
                        self.population[i] = new_population[i]
                        self.fitness[i] = new_fitness
                        if new_fitness < self.best_fitness:
                            self.best_fitness = new_fitness
                            self.best_position = self.population[i].copy()
                            self.stagnation_counter = 0 # Reset stagnation counter
                    else:
                        self.stagnation_counter += 1
            else:
                # Local Search with Nelder-Mead on the best individual
                
                res = minimize(func, self.best_position, method='Nelder-Mead', options={'maxiter': self.nm_max_iter, 'maxfev': self.budget - eval_count})
                
                if res.success:
                    if res.fun < self.best_fitness:
                        self.best_fitness = res.fun
                        self.best_position = res.x.copy()
                        self.stagnation_counter = 0
                        
                    eval_count += res.nfev # res.nfev is the number of function evaluations performed.

                else:
                    self.stagnation_counter += 1 # If Nelder-Mead fails, increase stagnation counter.

                #Perturb population after NM
                for i in range(self.pop_size):
                    self.population[i] = np.random.uniform(-5,5,self.dim)

            #Stagnation Check
            if self.stagnation_counter > self.stagnation_threshold:
                # Reset population to encourage exploration if stagnating
                self.population = np.random.uniform(-5, 5, size=(self.pop_size, self.dim))
                for i in range(self.pop_size):
                    if eval_count < self.budget:
                        self.fitness[i] = func(self.population[i])
                        eval_count += 1
                        if self.fitness[i] < self.best_fitness:
                            self.best_fitness = self.fitness[i]
                            self.best_position = self.population[i].copy()
                self.stagnation_counter = 0 # Reset the counter after re-initialization


        return self.best_fitness, self.best_position