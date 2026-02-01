import numpy as np

class DynamicPopulationDE:
    def __init__(self, budget=10000, dim=10, initial_pop_size=50, min_pop_size=10, max_pop_size=100, F=0.5, CR=0.7, adapt_prob=0.1):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = initial_pop_size
        self.min_pop_size = min_pop_size
        self.max_pop_size = max_pop_size
        self.F = F
        self.CR = CR
        self.adapt_prob = adapt_prob  # Probability of adapting F and CR
        self.population = None
        self.fitness = None
        self.success_history = None
        self.f_opt = np.inf
        self.x_opt = None

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.initial_pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.initial_pop_size
        self.success_history = np.zeros(self.initial_pop_size)  # Initialize success history for each individual
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)].copy()

    def adjust_population_size(self):
        """Adjust population size based on overall population success."""
        success_rate = np.mean(self.success_history)
        if success_rate > 0.6 and len(self.population) < self.max_pop_size:
            # Increase population size if success rate is high
            num_new = min(int(len(self.population) * 0.2), self.max_pop_size - len(self.population))
            if num_new > 0:
                new_individuals = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(num_new, self.dim))
                new_fitness = np.array([func(x) for x in new_individuals])
                self.budget -= num_new
                self.population = np.vstack((self.population, new_individuals))
                self.fitness = np.concatenate((self.fitness, new_fitness))
                self.success_history = np.concatenate((self.success_history, np.zeros(num_new))) # Initialize success history
        elif success_rate < 0.2 and len(self.population) > self.min_pop_size:
            # Decrease population size if success rate is low
            num_remove = min(int(len(self.population) * 0.2), len(self.population) - self.min_pop_size)
            if num_remove > 0:
                #Remove worst performing individuals
                worst_indices = np.argsort(self.fitness)[-num_remove:]
                keep_indices = np.setdiff1d(np.arange(len(self.population)), worst_indices)
                self.population = self.population[keep_indices]
                self.fitness = self.fitness[keep_indices]
                self.success_history = self.success_history[keep_indices]


    def __call__(self, func):
        self.initialize_population(func)

        while self.budget > 0:
            self.adjust_population_size()

            for i in range(len(self.population)):
                # Mutation
                indices = np.random.choice(len(self.population), 3, replace=False)
                x_r1, x_r2, x_r3 = self.population[indices]
                
                # Adapt F based on individual success history
                if np.random.rand() < self.adapt_prob:
                    self.F = 0.1 + 0.9 * self.success_history[i]

                v = self.population[i] + self.F * (x_r1 - x_r2) + self.F * (x_r3 - self.population[i])
                v = np.clip(v, func.bounds.lb, func.bounds.ub)

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
                    self.success_history[i] = 0.9 * self.success_history[i] + 0.1 # Update success history
                    self.fitness[i] = f_u
                    self.population[i] = u
                    if f_u < self.f_opt:
                        self.f_opt = f_u
                        self.x_opt = u.copy()
                else:
                    self.success_history[i] *= 0.9 # Decrease the success if the trial was not succesful

        return self.f_opt, self.x_opt