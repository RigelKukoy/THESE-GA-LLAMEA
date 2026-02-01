import numpy as np

class AgingDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, CR=0.9, F=0.5, age_limit=50):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.CR = CR
        self.F = F
        self.age_limit = age_limit
        self.population = None
        self.fitness = None
        self.ages = None

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.ages = np.zeros(self.pop_size, dtype=int)
        self.budget -= self.pop_size

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

                # Selection
                f_u_i = func(u_i)
                self.budget -= 1

                if f_u_i < self.fitness[i]:
                    self.population[i] = u_i
                    self.fitness[i] = f_u_i
                    self.ages[i] = 0  # Reset age
                    if f_u_i < self.f_opt:
                        self.f_opt = f_u_i
                        self.x_opt = u_i
                else:
                    self.ages[i] += 1  # Increment age

                # Aging: Replace old individuals
                if self.ages[i] > self.age_limit:
                    self.population[i] = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
                    self.fitness[i] = func(self.population[i])
                    self.ages[i] = 0
                    self.budget -= 1
                    if self.fitness[i] < self.f_opt:
                        self.f_opt = self.fitness[i]
                        self.x_opt = self.population[i]


            if self.budget <= 0:
                break

        return self.f_opt, self.x_opt