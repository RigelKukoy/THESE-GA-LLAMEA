import numpy as np

class DynamicAdaptiveDE:
    def __init__(self, budget=10000, dim=10, initial_pop_size=50, min_pop_size=10, CR=0.9, initial_F=0.5, F_decay=0.99, learning_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = initial_pop_size
        self.min_pop_size = min_pop_size
        self.CR = CR
        self.F = initial_F
        self.initial_F = initial_F
        self.F_decay = F_decay
        self.learning_rate = learning_rate
        self.population = None
        self.fitness = None
        self.best_fitness_history = []
        self.successful_F = []
        self.archive = []

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.best_fitness_history.append(np.min(self.fitness))
        self.budget -= self.pop_size

    def mutate(self, x_i):
        if np.random.rand() < 0.1: # Cauchy mutation for exploration
            return x_i + 0.01 * np.random.standard_cauchy(size=self.dim)

        indices = np.random.choice(self.pop_size, size=3, replace=False)
        x_r1 = self.population[indices[0]]
        x_r2 = self.population[indices[1]]
        x_r3 = self.population[indices[2]]

        if self.successful_F:
            F = np.random.choice(self.successful_F)
        else:
            F = self.F
        
        return x_r1 + F * (x_r2 - x_r3)

    def crossover(self, x_i, v_i):
        u_i = np.copy(x_i)
        j_rand = np.random.randint(self.dim)
        for j in range(self.dim):
            if np.random.rand() <= self.CR or j == j_rand:
                u_i[j] = v_i[j]
        return u_i

    def repair(self, x, func):
        return np.clip(x, func.bounds.lb, func.bounds.ub)

    def adjust_population_size(self):
        if len(self.best_fitness_history) > 2:
            improvement = self.best_fitness_history[-2] - self.best_fitness_history[-1]
            if improvement > 0:
                self.pop_size = min(self.pop_size + 1, 2 * self.min_pop_size) # Increase population if there is improvement, but with limit
            else:
                self.pop_size = max(self.pop_size - 1, self.min_pop_size) # Decrease population if no improvement

            # resize the population array by creating a new one and copying the old one
            new_population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
            new_population[:min(self.pop_size,len(self.population))] = self.population[:min(self.pop_size,len(self.population))]
            self.population = new_population
            
            new_fitness = np.array([func(x) for x in self.population])
            new_fitness[:min(len(new_fitness), len(self.fitness))] = self.fitness[:min(len(new_fitness), len(self.fitness))]
            self.fitness = new_fitness

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
                    self.successful_F.append(self.F)
                    if len(self.successful_F) > 10:
                        self.successful_F.pop(0)
                    
                    self.population[i] = u_i
                    self.fitness[i] = f_u_i
                    if f_u_i < self.f_opt:
                        self.f_opt = f_u_i
                        self.x_opt = u_i
                else:
                    self.archive.append(self.population[i])
                    if len(self.archive) > 2 * self.pop_size:
                        self.archive.pop(0)


            self.best_fitness_history.append(np.min(self.fitness))
            self.F *= self.F_decay # Decrease F gradually

            self.adjust_population_size()

            if self.budget <= 0:
                break

        return self.f_opt, self.x_opt