import numpy as np
from bayes_opt import BayesianOptimization

class AdaptiveDE_Archive_BayesLS:
    def __init__(self, budget=10000, dim=10, initial_pop_size=50, F=0.5, CR=0.9, archive_size=10, local_search_prob=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = initial_pop_size
        self.initial_pop_size = initial_pop_size
        self.F = F
        self.CR = CR
        self.archive_size = archive_size
        self.archive = []
        self.local_search_prob = local_search_prob

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

            # Update Archive
            for i in range(self.pop_size):
                if len(self.archive) < self.archive_size:
                    self.archive.append((self.population[i], fitness[i]))
                else:
                    worst_arch_idx = np.argmax([item[1] for item in self.archive]) #find worst fitness in archive
                    if fitness[i] < self.archive[worst_arch_idx][1]:
                        self.archive[worst_arch_idx] = (self.population[i], fitness[i])

            for i in range(self.pop_size):
                # Differential Evolution
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                x_r1, x_r2, x_r3 = self.population[idxs]
                
                #Use archive with a small probability
                if np.random.rand() < 0.1 and len(self.archive) > 0:
                    arch_idx = np.random.randint(0, len(self.archive))
                    x_r1 = self.archive[arch_idx][0]
                    
                mutant = x_r1 + self.F * (x_r2 - x_r3)
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)
                
                crossover = np.random.uniform(size=self.dim) < self.CR
                trial = np.where(crossover, mutant, self.population[i])
                
                # Local Search with Probability using Bayesian Optimization
                if np.random.rand() < self.local_search_prob:
                    # Define the bounds for the Bayesian Optimization
                    pbounds = {f'x{k}': (max(func.bounds.lb[k], trial[k] - 0.5), min(func.bounds.ub[k], trial[k] + 0.5)) for k in range(self.dim)}

                    def local_function( **kwargs):
                        x = np.array([kwargs[f'x{k}'] for k in range(self.dim)])
                        f = func(x)
                        return -f  #BO maximizes, so negate for minimization

                    optimizer = BayesianOptimization(
                        f=local_function,
                        pbounds=pbounds,
                        random_state=1,
                    )
                    
                    init_points = min(5, self.budget)  # Reduce if budget is low
                    n_iter = min(10, self.budget-init_points)  # Reduce if budget is low

                    optimizer.maximize(
                        init_points=init_points,
                        n_iter=n_iter,
                    )
                    
                    self.budget -= (init_points + n_iter)
                    
                    best_params = optimizer.max['params']
                    trial = np.array([best_params[f'x{k}'] for k in range(self.dim)])
                    

                # Selection
                f_trial = func(trial)
                self.budget -= 1
                
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial
                    
                if f_trial < fitness[i]:
                    new_fitness[i] = f_trial
                    new_population[i] = trial
                else:
                    new_fitness[i] = fitness[i]
                    new_population[i] = self.population[i]

            self.population = new_population
            fitness = new_fitness
            
            # Adjust population size dynamically
            if generation % 10 == 0:
                if np.mean(fitness) < np.mean([func(np.random.uniform(func.bounds.lb, func.bounds.ub)) for _ in range(100)]):
                    self.pop_size = min(2 * self.pop_size, self.initial_pop_size * 3) # Increase pop size if doing well
                    self.population = np.concatenate((self.population, np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size - len(self.population), self.dim))))
                    fitness = np.concatenate((fitness, np.array([func(x) for x in self.population[len(fitness):]])))
                    self.budget -= (self.pop_size - len(fitness))
                else:
                    self.pop_size = max(self.initial_pop_size, self.pop_size // 2) # Decrease pop size if not improving
                    self.population = self.population[:self.pop_size]
                    fitness = fitness[:self.pop_size]


        return self.f_opt, self.x_opt