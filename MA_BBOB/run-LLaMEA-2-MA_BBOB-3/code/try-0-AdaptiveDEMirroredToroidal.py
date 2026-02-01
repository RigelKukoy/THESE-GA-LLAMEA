import numpy as np

class AdaptiveDEMirroredToroidal:
    def __init__(self, budget=10000, dim=10, initial_pop_size=50, CR=0.9, initial_F=0.5, mirrored_rate=0.2, F_adapt_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = initial_pop_size
        self.CR = CR
        self.F = initial_F
        self.mirrored_rate = mirrored_rate
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

        return x_r1 + self.F * (x_r2 - x_r3)

    def crossover(self, x_i, v_i):
        u_i = np.copy(x_i)
        j_rand = np.random.randint(self.dim)
        for j in range(self.dim):
            if np.random.rand() <= self.CR or j == j_rand:
                u_i[j] = v_i[j]
        return u_i

    def repair_toroidal(self, x, func):
        """Handles boundaries using toroidal wrapping."""
        lb = func.bounds.lb
        ub = func.bounds.ub
        width = ub - lb
        x_wrapped = np.copy(x)
        for i in range(self.dim):
            if x_wrapped[i] < lb:
                x_wrapped[i] = ub - (lb - x_wrapped[i]) % width
            elif x_wrapped[i] > ub:
                x_wrapped[i] = lb + (x_wrapped[i] - ub) % width
        return x_wrapped

    def adapt_F(self, f_u_i, f_x_i):
        """Adapt F based on the fitness improvement."""
        if f_u_i < f_x_i:
            self.F = max(0.1, self.F * (1 - self.F_adapt_rate))  # Reduce F if improvement
        else:
            self.F = min(0.9, self.F * (1 + self.F_adapt_rate))  # Increase F if no improvement

    def mirrored_sampling(self, func):
        """Performs mirrored sampling around the best solution."""
        num_mirrored = int(self.mirrored_rate * self.pop_size)
        if num_mirrored > 0:
            best_index = np.argmin(self.fitness)
            x_best = self.population[best_index]
            lb = func.bounds.lb
            ub = func.bounds.ub
            for _ in range(num_mirrored):
                x_mirrored = x_best + np.random.uniform(-0.1 * (ub - lb), 0.1 * (ub - lb), size=self.dim) # Local exploration
                x_mirrored = self.repair_toroidal(x_mirrored, func)

                f_mirrored = func(x_mirrored)
                self.budget -= 1
                
                worst_index = np.argmax(self.fitness)
                if f_mirrored < self.fitness[worst_index]:
                        self.population[worst_index] = x_mirrored
                        self.fitness[worst_index] = f_mirrored
                        if f_mirrored < self.f_opt:
                            self.f_opt = f_mirrored
                            self.x_opt = x_mirrored


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

                # Repair (Toroidal)
                u_i = self.repair_toroidal(u_i, func)

                # Selection
                f_u_i = func(u_i)
                self.budget -= 1

                if f_u_i < self.fitness[i]:
                    self.population[i] = u_i
                    self.fitness[i] = f_u_i
                    if f_u_i < self.f_opt:
                        self.f_opt = f_u_i
                        self.x_opt = u_i
                    self.adapt_F(f_u_i, self.fitness[i])

            self.mirrored_sampling(func) #Mirrored sampling
            
            self.best_fitness_history.append(np.min(self.fitness))
            if self.budget <= 0:
                break

        return self.f_opt, self.x_opt