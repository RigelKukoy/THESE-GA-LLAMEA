import numpy as np

class AdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size_init=50, F_init=0.5, CR_init=0.7, F_lr=0.1, CR_lr=0.1, stagnation_threshold=100, local_search_iterations=100):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size_init
        self.pop_size_min = 10  # Minimum population size
        self.pop_size_max = 100 # Maximum population size
        self.F = F_init
        self.CR = CR_init
        self.F_lr = F_lr
        self.CR_lr = CR_lr
        self.stagnation_threshold = stagnation_threshold
        self.local_search_iterations = local_search_iterations
        self.population = None
        self.fitness = None
        self.best_index = None
        self.f_opt = np.inf
        self.x_opt = None
        self.success_history_F = []
        self.success_history_CR = []
        self.stagnation_counter = 0
        self.last_improvement = 0

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        self.best_index = np.argmin(self.fitness)
        self.f_opt = self.fitness[self.best_index]
        self.x_opt = self.population[self.best_index].copy()
        self.last_improvement = 0
        self.stagnation_counter = 0

    def mutation(self, i):
        indices = np.random.choice(self.pop_size, 3, replace=False)
        x_r1, x_r2, x_r3 = self.population[indices]
        return self.population[i] + self.F * (x_r2 - x_r3)

    def crossover(self, v, i):
        j_rand = np.random.randint(self.dim)
        u = self.population[i].copy()
        for j in range(self.dim):
            if np.random.rand() < self.CR or j == j_rand:
                u[j] = v[j]
        return u

    def local_search(self, func):
        # Perform a simple gradient-based local search around the best solution
        learning_rate = 0.1
        x_current = self.x_opt.copy()
        f_current = self.f_opt
        for _ in range(self.local_search_iterations):
            gradient = np.zeros_like(x_current)
            for j in range(self.dim):
                x_plus = x_current.copy()
                x_minus = x_current.copy()
                delta = 1e-5
                x_plus[j] += delta
                x_minus[j] -= delta
                x_plus = np.clip(x_plus, func.bounds.lb, func.bounds.ub)
                x_minus = np.clip(x_minus, func.bounds.lb, func.bounds.ub)
                f_plus = func(x_plus)
                f_minus = func(x_minus)
                self.budget -= 2
                gradient[j] = (f_plus - f_minus) / (2 * delta)
                if self.budget <= 0:
                    return

            x_new = x_current - learning_rate * gradient
            x_new = np.clip(x_new, func.bounds.lb, func.bounds.ub)
            f_new = func(x_new)
            self.budget -= 1

            if f_new < f_current:
                f_current = f_new
                x_current = x_new
                self.f_opt = f_new
                self.x_opt = x_new.copy()


    def adjust_population_size(self):
        # Dynamically adjust population size based on stagnation
        if self.stagnation_counter > self.stagnation_threshold:
            self.pop_size = max(self.pop_size // 2, self.pop_size_min)
            print(f"Stagnation detected. Reducing population size to {self.pop_size}")
            self.stagnation_counter = 0
            # Reinitialize population
            #self.initialize_population(func) # Reinitialization should be done outside of this function
        elif self.stagnation_counter < self.stagnation_threshold / 2 and self.pop_size < self.pop_size_max:
            self.pop_size = min(self.pop_size * 2, self.pop_size_max)
            print(f"Improving. Increasing population size to {self.pop_size}")
            #self.initialize_population(func)

    def __call__(self, func):
        self.initialize_population(func)

        while self.budget > 0:
            
            # Stagnation Detection
            if self.fitness[self.best_index] == self.f_opt:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
                self.best_index = np.argmin(self.fitness)
                self.f_opt = self.fitness[self.best_index]
                self.x_opt = self.population[self.best_index].copy()

            if self.stagnation_counter > self.stagnation_threshold:
                print("Local Search Triggered")
                self.local_search(func)
                self.stagnation_counter = 0

            self.adjust_population_size()

            # Reinitialize population with the new size
            new_population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
            new_fitness = np.array([func(x) for x in new_population])
            self.budget -= self.pop_size
            
            # Keep best individual from the old population
            new_population[0] = self.x_opt.copy()
            new_fitness[0] = self.f_opt

            self.population = new_population
            self.fitness = new_fitness

            for i in range(self.pop_size):
                # Parameter adaptation using success history
                if self.success_history_F:
                    self.F = np.random.choice(self.success_history_F)
                if self.success_history_CR:
                    self.CR = np.random.choice(self.success_history_CR)

                # Mutation
                v = self.mutation(i)
                v = np.clip(v, func.bounds.lb, func.bounds.ub)

                # Crossover
                u = self.crossover(v, i)

                # Evaluation
                f_u = func(u)
                self.budget -= 1

                # Selection
                if f_u < self.fitness[i]:
                    self.success_history_F.append(self.F)
                    self.success_history_CR.append(self.CR)

                    self.fitness[i] = f_u
                    self.population[i] = u

                    if f_u < self.f_opt:
                        self.f_opt = f_u
                        self.x_opt = u.copy()
                        self.stagnation_counter = 0 #reset stagnation counter

                # Limit the size of the success history
                self.success_history_F = self.success_history_F[-10:]
                self.success_history_CR = self.success_history_CR[-10:]

                if self.budget <= 0:
                    break

        return self.f_opt, self.x_opt