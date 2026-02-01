import numpy as np

class MirroredSamplingDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.5, CR=0.9, mirror_rate=0.2):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.mirror_rate = mirror_rate
        self.population = None
        self.fitness = None

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        
    def mirrored_sample(self, x, func):
        """Generate a mirrored sample within the bounds."""
        lb = func.bounds.lb
        ub = func.bounds.ub
        
        mirrored_x = x + self.mirror_rate * (np.random.rand(self.dim) * (ub - lb) - (x - lb))
        mirrored_x = np.clip(mirrored_x, lb, ub)
        return mirrored_x
        

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

                # Mirrored Sample
                if np.random.rand() < self.mirror_rate:
                  u_i = self.mirrored_sample(u_i, func)
                

                # Selection
                f_u_i = func(u_i)
                self.budget -= 1
                if f_u_i < self.fitness[i]:
                    self.population[i] = u_i
                    self.fitness[i] = f_u_i

                    if f_u_i < self.f_opt:
                        self.f_opt = f_u_i
                        self.x_opt = u_i
                
                if self.budget <= 0:
                  break

        return self.f_opt, self.x_opt