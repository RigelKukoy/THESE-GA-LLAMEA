import numpy as np

class SelfAdaptivePopulationDE:
    def __init__(self, budget=10000, dim=10, initial_pop_size=50, F=0.5, CR=0.7, adaptive_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = initial_pop_size
        self.initial_pop_size = initial_pop_size
        self.F = F
        self.CR = CR
        self.adaptive_rate = adaptive_rate
        self.population = None
        self.fitness = None
        self.best_index = None
        self.f_opt = np.inf
        self.x_opt = None
        self.generation = 0

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        self.best_index = np.argmin(self.fitness)
        self.f_opt = self.fitness[self.best_index]
        self.x_opt = self.population[self.best_index].copy()

    def orthogonal_learning(self, func, x):
        # Orthogonal array design for local search around x
        levels = 3  # Number of levels for each dimension
        orthogonal_array = np.array([
            [-1, -1],
            [-1, 1],
            [1, -1],
            [1, 1],
            [0,0]
        ])
        
        best_f = np.inf
        best_x = None
        
        for i in range(orthogonal_array.shape[0]):
            new_x = x.copy()
            step_size = 0.05 # Small step size
            for j in range(min(self.dim, orthogonal_array.shape[1])):
                 new_x[j] = x[j] + orthogonal_array[i, j] * step_size
                 new_x = np.clip(new_x, func.bounds.lb, func.bounds.ub)
            f = func(new_x)
            self.budget -= 1
            if f < best_f:
                best_f = f
                best_x = new_x
        
        return best_f, best_x

    def adjust_population_size(self):
        # Dynamically adjust population size based on stagnation
        if self.generation % 10 == 0:
            if self.f_opt == self.f_opt_prev:
                # Stagnation detected: Increase population size
                self.pop_size = min(int(self.pop_size * (1 + self.adaptive_rate)), 2 * self.initial_pop_size)
            else:
                # Improvement: Decrease population size
                self.pop_size = max(int(self.pop_size * (1 - self.adaptive_rate)), self.initial_pop_size // 2)
            
            # Ensure population size remains within bounds and is an integer
            self.pop_size = max(10, min(self.pop_size, 100))
            self.pop_size = int(self.pop_size)

            # Resize population (crude, but effective) - reinitialize
            if self.pop_size != self.population.shape[0]:
                self.initialize_population(self.func_ref) # Reinitialize including fitness calculation


    def __call__(self, func):
        self.func_ref = func
        self.initialize_population(func)
        self.f_opt_prev = np.inf

        while self.budget > 0:
            self.generation += 1
            self.adjust_population_size()

            for i in range(self.pop_size):
                # Mutation
                indices = np.random.choice(self.pop_size, 3, replace=False)
                x_r1, x_r2, x_r3 = self.population[indices]
                v = self.population[i] + self.F * (x_r1 - x_r2)
                v = np.clip(v, func.bounds.lb, func.bounds.ub)

                # Crossover
                j_rand = np.random.randint(self.dim)
                u = self.population[i].copy()
                for j in range(self.dim):
                    if np.random.rand() < self.CR or j == j_rand:
                        u[j] = v[j]
                
                u = np.clip(u, func.bounds.lb, func.bounds.ub)

                # Evaluation
                f_u = func(u)
                self.budget -= 1

                # Selection
                if f_u < self.fitness[i]:
                    self.fitness[i] = f_u
                    self.population[i] = u
                    if f_u < self.f_opt:
                        self.f_opt_prev = self.f_opt
                        self.f_opt = f_u
                        self.x_opt = u.copy()
                        #Orthogonal learning around the new best
                        f_ortho, x_ortho = self.orthogonal_learning(func, self.x_opt)
                        if f_ortho < self.f_opt:
                            self.f_opt = f_ortho
                            self.x_opt = x_ortho.copy()
                            

            
        return self.f_opt, self.x_opt