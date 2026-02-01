import numpy as np
from scipy.optimize import minimize

class AdaptiveDESimplexLocal:
    def __init__(self, budget=10000, dim=10, pop_size=50, CR=0.9, F_init=0.5, local_search_prob=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.CR = CR
        self.F_init = F_init  # Initial scaling factor
        self.F = F_init
        self.local_search_prob = local_search_prob
        self.population = None
        self.fitness = None
        self.f_opt = np.Inf
        self.x_opt = None

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]
        self.budget -= self.pop_size

    def mutate(self, x_i):
        indices = np.random.choice(self.pop_size, size=3, replace=False)
        x_r1 = self.population[indices[0]]
        x_r2 = self.population[indices[1]]
        x_r3 = self.population[indices[2]]

        # Fitness-dependent scaling factor
        delta_fitness = np.abs(self.fitness[indices[0]] - self.fitness[indices[1]] + self.fitness[indices[2]])/3
        self.F = self.F_init * (1 + delta_fitness)  # Adjust scaling factor based on fitness differences.
        self.F = np.clip(self.F, 0.1, 1.0)

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

    def local_search(self, x, func):
        bounds = [(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)]
        
        def obj_func(x_):
            val = func(x_)
            self.budget -= 1
            return val

        result = minimize(obj_func, x, method='Nelder-Mead', bounds=bounds, options={'maxfev': min(50, self.budget)})  # Limited local search
        if result.success:
            return result.x, result.fun
        else:
            return x, func(x)
    

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
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
                    self.population[i] = u_i
                    self.fitness[i] = f_u_i

                    if f_u_i < self.f_opt:
                        self.f_opt = f_u_i
                        self.x_opt = u_i

            # Local search around the best solution
            if np.random.rand() < self.local_search_prob and self.budget > 0:
                x_new, f_new = self.local_search(self.x_opt, func)
                if f_new < self.f_opt:
                    self.f_opt = f_new
                    self.x_opt = x_new

            if self.budget <= 0:
                break

        return self.f_opt, self.x_opt