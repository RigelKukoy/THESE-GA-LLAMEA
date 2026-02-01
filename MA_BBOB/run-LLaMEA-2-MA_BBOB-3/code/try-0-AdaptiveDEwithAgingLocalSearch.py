import numpy as np

class AdaptiveDEwithAgingLocalSearch:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.5, CR=0.9, age_limit=50, local_search_prob=0.1, local_search_radius=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.age_limit = age_limit
        self.local_search_prob = local_search_prob
        self.local_search_radius = local_search_radius
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

    def local_search(self, x, func):
        """Perform a local search around the given solution."""
        new_x = x + np.random.uniform(-self.local_search_radius, self.local_search_radius, size=self.dim)
        new_x = self.repair(new_x, func)
        new_f = func(new_x)
        self.budget -= 1
        return new_x, new_f

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

                # Aging penalty
                f_i = self.fitness[i] + (self.ages[i] / self.age_limit)  # Penalize older individuals
                
                if f_u_i < f_i:
                    self.population[i] = u_i
                    self.fitness[i] = f_u_i
                    self.ages[i] = 0  # Reset age
                    if f_u_i < self.f_opt:
                        self.f_opt = f_u_i
                        self.x_opt = u_i
                else:
                    self.ages[i] += 1

                # Local Search
                if np.random.rand() < self.local_search_prob:
                    new_x, new_f = self.local_search(self.population[i], func)
                    if new_f < self.fitness[i]:
                        self.population[i] = new_x
                        self.fitness[i] = new_f
                        self.ages[i] = 0
                        if new_f < self.f_opt:
                            self.f_opt = new_f
                            self.x_opt = new_x
                
                if self.budget <= 0:
                    break
            self.ages += 1 # Increment ages for all individuals
        return self.f_opt, self.x_opt