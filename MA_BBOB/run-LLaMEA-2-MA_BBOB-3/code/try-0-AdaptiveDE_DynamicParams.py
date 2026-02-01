import numpy as np

class AdaptiveDE_DynamicParams:
    def __init__(self, budget=10000, dim=10, pop_size=50, archive_size=100, F_init=0.5, CR_init=0.9, F_adapt_rate=0.1, CR_adapt_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.archive_size = archive_size
        self.F = F_init
        self.CR = CR_init
        self.F_init = F_init
        self.CR_init = CR_init
        self.F_adapt_rate = F_adapt_rate
        self.CR_adapt_rate = CR_adapt_rate
        self.population = None
        self.fitness = None
        self.archive = None
        self.success_F = []
        self.success_CR = []

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
        self.F = self.F_init
        self.CR = self.CR_init

    def adapt_parameters(self):
        if len(self.success_F) > 0:
            self.F = np.mean(self.success_F)
            self.F = np.clip(self.F, 0.1, 0.9)  # Keep F within reasonable bounds
        else:
            self.F = self.F_init
        if len(self.success_CR) > 0:
            self.CR = np.mean(self.success_CR)
            self.CR = np.clip(self.CR, 0.1, 0.9) # Keep CR within reasonable bounds
        else:
            self.CR = self.CR_init
        self.success_F = []
        self.success_CR = []
    
    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        self.initialize_population(func)

        iteration = 0
        while self.budget > 0:
            iteration += 1
            
            successful_F = []
            successful_CR = []

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
                    successful_F.append(self.F)
                    successful_CR.append(self.CR)

                    self.population[i] = u_i
                    self.fitness[i] = f_u_i

                    if f_u_i < self.f_opt:
                        self.f_opt = f_u_i
                        self.x_opt = u_i

            self.update_archive()
            self.success_F.extend(successful_F)
            self.success_CR.extend(successful_CR)
            self.adapt_parameters()


            # Restart mechanism (every 100 iterations)
            if iteration % 100 == 0 and self.budget > self.pop_size:
                self.restart(func)

        return self.f_opt, self.x_opt