import numpy as np

class AdaptiveDEGaussianRestart:
    def __init__(self, budget=10000, dim=10, pop_size=50, CR=0.9, F_init=0.5, lr_F=0.1, restart_patience=50):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.CR = CR
        self.F = F_init
        self.lr_F = lr_F
        self.population = None
        self.fitness = None
        self.f_opt = np.Inf
        self.x_opt = None
        self.restart_patience = restart_patience
        self.stagnation_counter = 0
        self.best_fitness_history = []

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]
        self.budget -= self.pop_size
        self.best_fitness_history.append(self.f_opt)

    def mutate(self, x_i):
        indices = np.random.choice(self.pop_size, size=3, replace=False)
        x_r1 = self.population[indices[0]]
        x_r2 = self.population[indices[1]]
        x_r3 = self.population[indices[2]]
        # Gaussian mutation with self-adaptive F
        return x_r1 + self.F * (x_r2 - x_r3) + np.random.normal(0, 0.01, size=self.dim)


    def crossover(self, x_i, v_i):
        u_i = np.copy(x_i)
        j_rand = np.random.randint(self.dim)
        for j in range(self.dim):
            if np.random.rand() <= self.CR or j == j_rand:
                u_i[j] = v_i[j]
        return u_i

    def repair(self, x, func):
        return np.clip(x, func.bounds.lb, func.bounds.ub)

    def adjust_learning_rate(self):
        if len(self.best_fitness_history) < 2:
            return

        if self.best_fitness_history[-1] < self.best_fitness_history[-2]:
            self.F = max(0, self.F * (1 + self.lr_F))  # Increase if improving
        else:
            self.F = max(0, self.F * (1 - self.lr_F))  # Decrease if stagnating

    def restart_population(self, func):
        # Re-initialize population around the current best solution with some noise.
        self.population = np.random.normal(loc=self.x_opt, scale=0.1, size=(self.pop_size, self.dim))
        for i in range(self.pop_size):
            self.population[i] = self.repair(self.population[i], func)
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        
        # Update best fitness if any of the new solutions are better
        current_best_fitness = np.min(self.fitness)
        if current_best_fitness < self.f_opt:
          self.f_opt = current_best_fitness
          self.x_opt = self.population[np.argmin(self.fitness)]

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        self.initialize_population(func)

        while self.budget > 0:
            
            # Stagnation check and restart
            if len(self.best_fitness_history) > 1 and self.best_fitness_history[-1] >= self.best_fitness_history[-2]:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
            
            if self.stagnation_counter >= self.restart_patience:
                self.restart_population(func)
                self.stagnation_counter = 0
            

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

            self.best_fitness_history.append(self.f_opt)
            self.adjust_learning_rate()

            if self.budget <= 0:
                break

        return self.f_opt, self.x_opt