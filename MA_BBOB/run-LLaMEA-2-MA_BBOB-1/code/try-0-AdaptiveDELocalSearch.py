import numpy as np

class AdaptiveDELocalSearch:
    def __init__(self, budget=10000, dim=10, pop_size=50, initial_F=0.5, initial_CR=0.7, stagnation_threshold=100, diversity_threshold=0.1, local_search_probability=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = initial_F
        self.CR = initial_CR
        self.stagnation_threshold = stagnation_threshold
        self.diversity_threshold = diversity_threshold
        self.local_search_probability = local_search_probability
        self.population = None
        self.fitness = None
        self.best_index = None
        self.f_opt = np.inf
        self.x_opt = None
        self.stagnation_counter = 0
        self.lb = None
        self.ub = None

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        self.best_index = np.argmin(self.fitness)
        self.f_opt = self.fitness[self.best_index]
        self.x_opt = self.population[self.best_index].copy()
        self.lb = func.bounds.lb
        self.ub = func.bounds.ub

    def calculate_diversity(self):
        """Calculates the average distance of each individual from the population mean."""
        mean_position = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - mean_position, axis=1)
        return np.mean(distances)

    def adjust_parameters(self):
        """Adjusts F and CR adaptively."""
        if np.random.rand() < 0.1:  # Adjust parameters with a probability
            self.F = np.random.uniform(0.3, 0.9)
            self.CR = np.random.uniform(0.1, 0.9)


    def local_search(self, func, x, radius=0.1):
         """Performs a simple local search around a given solution."""
         x_new = x.copy()
         for i in range(self.dim):
             delta = np.random.uniform(-radius, radius)
             x_new[i] = np.clip(x[i] + delta, self.lb, self.ub)
         f_new = func(x_new)
         self.budget -=1
         return f_new, x_new

    def check_stagnation(self):
        """Checks if the optimization is stagnating."""
        current_best_fitness = self.fitness[self.best_index]
        if current_best_fitness >= self.f_opt:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
            self.f_opt = current_best_fitness

        if self.stagnation_counter > self.stagnation_threshold:
            return True
        else:
            return False

    def restart_population(self, func):
        """Restarts the population if diversity is too low."""
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        self.best_index = np.argmin(self.fitness)
        self.f_opt = self.fitness[self.best_index]
        self.x_opt = self.population[self.best_index].copy()
        self.stagnation_counter = 0  # Reset stagnation counter



    def __call__(self, func):
        self.initialize_population(func)

        while self.budget > 0:
            self.adjust_parameters()

            for i in range(self.pop_size):
                # Mutation
                indices = np.random.choice(self.pop_size, 3, replace=False)
                x_r1, x_r2, x_r3 = self.population[indices]
                v = self.population[i] + self.F * (x_r2 - x_r3)
                v = np.clip(v, self.lb, self.ub)

                # Crossover
                j_rand = np.random.randint(self.dim)
                u = self.population[i].copy()
                for j in range(self.dim):
                    if np.random.rand() < self.CR or j == j_rand:
                        u[j] = v[j]

                # Evaluation
                f_u = func(u)
                self.budget -= 1

                # Selection
                if f_u < self.fitness[i]:
                    self.fitness[i] = f_u
                    self.population[i] = u
                    if f_u < self.f_opt:
                        self.f_opt = f_u
                        self.x_opt = u.copy()
                        self.best_index = np.argmin(self.fitness)
            
            # Local Search
            if np.random.rand() < self.local_search_probability:
               f_ls, x_ls = self.local_search(func, self.x_opt)
               if f_ls < self.f_opt:
                   self.f_opt = f_ls
                   self.x_opt = x_ls.copy()
                   self.fitness[self.best_index] = f_ls
                   self.population[self.best_index] = x_ls.copy()


            # Stagnation Check
            if self.check_stagnation():
                diversity = self.calculate_diversity()
                if diversity < self.diversity_threshold:
                    self.restart_population(func)
                else:
                    #If stagnation is detected, but diversity is still good, we increase exploration
                    self.F = np.clip(self.F*1.2, 0.1, 0.9)
                    self.CR = np.clip(self.CR*0.8, 0.1, 0.9)


        return self.f_opt, self.x_opt