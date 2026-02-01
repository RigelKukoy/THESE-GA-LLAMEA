import numpy as np

class AdaptiveDERestart:
    def __init__(self, budget=10000, dim=10, pop_size=50, CR=0.9, stagnation_limit=100, F_init=0.5, F_adapt_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.CR = CR
        self.stagnation_limit = stagnation_limit
        self.F = F_init
        self.F_adapt_rate = F_adapt_rate
        self.population = None
        self.fitness = None
        self.best_fitness_history = []
        self.stagnation_counter = 0

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]
        self.best_fitness_history.append(self.f_opt)

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

    def adjust_mutation_factor(self):
        """Adjust F based on population diversity."""
        diversity = np.std(self.fitness)
        if diversity > 0:
            self.F = np.clip(self.F + self.F_adapt_rate * (0.5 - np.random.rand()), 0.1, 1.0)  # Adapt F based on diversity
        else:
            self.F = 0.5

    def check_stagnation(self):
        """Check if the optimization has stagnated."""
        if len(self.best_fitness_history) > self.stagnation_limit:
            if np.abs(self.best_fitness_history[-1] - np.mean(self.best_fitness_history[-self.stagnation_limit:])) < 1e-6:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0

        if self.stagnation_counter >= self.stagnation_limit:
            return True
        else:
            return False

    def restart_population(self, func):
        """Restart the population with new random individuals."""
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]
        self.best_fitness_history.append(self.f_opt)
        self.stagnation_counter = 0

    def __call__(self, func):
        self.initialize_population(func)

        while self.budget > 0:
            self.adjust_mutation_factor()

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

            best_fitness = np.min(self.fitness)
            self.best_fitness_history.append(best_fitness)

            if best_fitness < self.f_opt:
                self.f_opt = best_fitness
                self.x_opt = self.population[np.argmin(self.fitness)]

            if self.check_stagnation():
                self.restart_population(func)

            if self.budget <= 0:
                break

        return self.f_opt, self.x_opt