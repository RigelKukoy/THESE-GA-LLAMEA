import numpy as np

class AdaptiveDERejuvenation:
    def __init__(self, budget=10000, dim=10, initial_pop_size=50, CR=0.9, initial_F=0.5, rejuvenation_rate=0.05, cauchy_scale=0.1, F_adapt_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = initial_pop_size
        self.CR = CR
        self.F = initial_F
        self.rejuvenation_rate = rejuvenation_rate
        self.cauchy_scale = cauchy_scale
        self.F_adapt_rate = F_adapt_rate
        self.population = None
        self.fitness = None
        self.best_fitness_history = []

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.best_fitness_history.append(np.min(self.fitness))
        self.budget -= self.pop_size

    def mutate(self, x_i):
        indices = np.random.choice(self.pop_size, size=3, replace=False)
        x_r1 = self.population[indices[0]]
        x_r2 = self.population[indices[1]]
        x_r3 = self.population[indices[2]]

        # Modified Cauchy mutation: adaptively scale Cauchy noise
        cauchy_noise = np.random.standard_cauchy(size=self.dim) * self.cauchy_scale * self.F
        return x_r1 + self.F * (x_r2 - x_r3) + cauchy_noise

    def crossover(self, x_i, v_i):
        u_i = np.copy(x_i)
        j_rand = np.random.randint(self.dim)
        for j in range(self.dim):
            if np.random.rand() <= self.CR or j == j_rand:
                u_i[j] = v_i[j]
        return u_i

    def repair(self, x, func):
        return np.clip(x, func.bounds.lb, func.bounds.ub)
    
    def adapt_F(self, f_u_i, f_x_i):
        """Adapt F based on the fitness improvement."""
        if f_u_i < f_x_i:
            self.F = max(0.1, self.F * (1 - self.F_adapt_rate))  # Reduce F if improvement
        else:
            self.F = min(0.9, self.F * (1 + self.F_adapt_rate))  # Increase F if no improvement
    

    def rejuvenate_population(self, func):
        """Replaces the worst individuals with new random ones."""
        num_rejuvenate = int(self.rejuvenation_rate * self.pop_size)
        if num_rejuvenate > 0:
            worst_indices = np.argsort(self.fitness)[-num_rejuvenate:]  # Indices of worst individuals
            for i in worst_indices:
                self.population[i] = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
                self.fitness[i] = func(self.population[i])
                self.budget -= 1

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
                    self.adapt_F(f_u_i, self.fitness[i]) #Adapt F value if improved

            self.rejuvenate_population(func)  # Rejuvenate population

            self.best_fitness_history.append(np.min(self.fitness))
            if self.budget <= 0:
                break

        return self.f_opt, self.x_opt