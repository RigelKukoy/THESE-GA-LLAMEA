import numpy as np

class AdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, archive_size=100, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.archive_size = archive_size
        self.F = F
        self.CR = CR
        self.population = None
        self.fitness = None
        self.archive = None

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        self.archive = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.archive_size, self.dim))
        
    def mutate(self, x_i, func):
        indices = np.random.choice(self.pop_size + self.archive_size, size=3, replace=False)
        
        if indices[0] < self.pop_size:
            x_r1 = self.population[indices[0]]
        else:
            x_r1 = self.archive[indices[0] - self.pop_size]

        if indices[1] < self.pop_size:
            x_r2 = self.population[indices[1]]
        else:
            x_r2 = self.archive[indices[1] - self.pop_size]
            
        if indices[2] < self.pop_size:
            x_r3 = self.population[indices[2]]
        else:
            x_r3 = self.archive[indices[2] - self.pop_size]

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

    def update_archive(self):
        # Randomly replace archive members with population members
        indices = np.random.choice(self.archive_size, size=self.pop_size, replace=False)
        self.archive[indices] = self.population

    def restart(self, func):
        # Restart the population with new random solutions
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
    
    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        self.initialize_population(func)

        iteration = 0
        while self.budget > 0:
            iteration += 1
            for i in range(self.pop_size):
                # Mutation
                v_i = self.mutate(self.population[i], func)
                
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

            self.update_archive()

            # Restart mechanism (every 100 iterations)
            if iteration % 100 == 0 and self.budget > self.pop_size:
                self.restart(func)

        return self.f_opt, self.x_opt