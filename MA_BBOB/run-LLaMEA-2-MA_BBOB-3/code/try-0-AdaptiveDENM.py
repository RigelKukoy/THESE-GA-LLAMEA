import numpy as np
from scipy.optimize import minimize

class AdaptiveDENM:
    def __init__(self, budget=10000, dim=10, pop_size=50, initial_F=0.5, initial_CR=0.9, F_decay=0.99, CR_decay=0.99, local_search_prob=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = initial_F
        self.CR = initial_CR
        self.initial_F = initial_F
        self.initial_CR = initial_CR
        self.F_decay = F_decay
        self.CR_decay = CR_decay
        self.local_search_prob = local_search_prob
        self.population = None
        self.fitness = None
        self.success_F = []
        self.success_CR = []
        self.memory_size = 10
        self.best_fitness = np.inf
        self.best_individual = None

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        self.best_fitness = np.min(self.fitness)
        self.best_individual = self.population[np.argmin(self.fitness)]

    def mutate(self, x_i):
        indices = np.random.choice(self.pop_size, size=3, replace=False)
        x_r1 = self.population[indices[0]]
        x_r2 = self.population[indices[1]]
        x_r3 = self.population[indices[2]]
        return x_r1 + self.F * (x_r2 - x_r3)

    def crossover(self, x_i, v_i):
        u_i = np.copy(x_i)
        j_rand = np.random.randint(self.dim)
        for j in range(self.dim):
            if np.random.rand() <= self.CR or j == j_rand:
                u_i[j] = v_i[j]
        return u_i

    def repair(self, x, func):
        return np.clip(x, func.bounds.lb, func.bounds.ub)
    
    def adapt_parameters(self):
        if self.success_F:
            self.F = np.mean(self.success_F)
            self.CR = np.mean(self.success_CR)
        self.F = self.F * self.F_decay
        self.CR = self.CR * self.CR_decay
        self.F = np.clip(self.F, 0.1, 1.0)
        self.CR = np.clip(self.CR, 0.1, 1.0)

        self.success_F = []
        self.success_CR = []

    def local_search(self, individual, func):
        bounds = [(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)]
        res = minimize(func, individual, method='Nelder-Mead', bounds=bounds, options={'maxfev': min(50, self.budget)})
        self.budget -= res.nfev
        return res.fun, res.x
    
    def __call__(self, func):
        self.initialize_population(func)
        
        while self.budget > 0:
            for i in range(self.pop_size):
                # Mutation
                v_i = self.mutate(self.population[i])

                # Crossover
                u_i = self.crossover(self.population[i], v_i)

                # Repair
                u_i = self.repair(u_i, func)

                # Selection
                f_u_i = func(u_i)
                self.budget -= 1
                
                if f_u_i < self.fitness[i]:
                    self.success_F.append(self.F)
                    self.success_CR.append(self.CR)
                    self.population[i] = u_i
                    self.fitness[i] = f_u_i

                    if f_u_i < self.best_fitness:
                        self.best_fitness = f_u_i
                        self.best_individual = u_i
                        
                # Local Search
                if np.random.rand() < self.local_search_prob and self.budget > 0:
                    f_local, x_local = self.local_search(self.population[i], func)
                    if f_local < self.fitness[i]:
                        self.population[i] = x_local
                        self.fitness[i] = f_local
                        if f_local < self.best_fitness:
                            self.best_fitness = f_local
                            self.best_individual = x_local
            
            self.adapt_parameters()

            if self.budget <= 0:
                break

        return self.best_fitness, self.best_individual