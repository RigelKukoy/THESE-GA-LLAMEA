import numpy as np

class AdaptiveDECauchyRestart:
    def __init__(self, budget=10000, dim=10, initial_pop_size=50, CR=0.9, initial_F=0.5, pop_size_reduction_factor=0.9, stagnation_limit=100, cauchy_scale=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = initial_pop_size
        self.initial_pop_size = initial_pop_size
        self.CR = CR
        self.F = initial_F
        self.pop_size_reduction_factor = pop_size_reduction_factor
        self.stagnation_limit = stagnation_limit
        self.cauchy_scale = cauchy_scale
        self.population = None
        self.fitness = None
        self.best_fitness_history = []
        self.stagnation_counter = 0
        self.entropy_history = []

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

        # Cauchy mutation
        cauchy_noise = np.random.standard_cauchy(size=self.dim) * self.cauchy_scale
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

    def calculate_fitness_entropy(self):
        """Calculates the entropy of the fitness values."""
        fitness_min = np.min(self.fitness)
        fitness_max = np.max(self.fitness)
        if fitness_max == fitness_min:
            return 0  # Avoid division by zero if all fitness values are the same

        normalized_fitness = (self.fitness - fitness_min) / (fitness_max - fitness_min)
        probabilities = normalized_fitness / np.sum(normalized_fitness)
        probabilities = probabilities[probabilities > 0] # Avoid log(0)
        entropy = -np.sum(probabilities * np.log(probabilities))
        return entropy

    def check_stagnation(self):
        """Checks for stagnation based on fitness entropy."""
        entropy = self.calculate_fitness_entropy()
        self.entropy_history.append(entropy)

        if len(self.entropy_history) > self.stagnation_limit:
            entropy_diff = np.abs(self.entropy_history[-1] - np.mean(self.entropy_history[-self.stagnation_limit:]))
            if  entropy_diff < 1e-6: #Stagnation if entropy doesn't change
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0

        if self.stagnation_counter >= self.stagnation_limit:
            return True
        else:
            return False
        
    def reduce_population_size(self):
        """Reduces the population size."""
        self.pop_size = int(self.pop_size * self.pop_size_reduction_factor)
        self.pop_size = max(10, self.pop_size)  # Ensure a minimum population size
        
        #Select top individuals
        top_indices = np.argsort(self.fitness)[:self.pop_size]
        self.population = self.population[top_indices]
        self.fitness = self.fitness[top_indices]
        

    def restart(self, func):
        """Restart the population with new random individuals."""
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.initial_pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.pop_size = self.initial_pop_size
        self.stagnation_counter = 0
        self.best_fitness_history = [np.min(self.fitness)]
        self.entropy_history = []
        self.budget -= self.initial_pop_size

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

            # Stagnation Check and Restart
            if self.check_stagnation():
                 if self.pop_size > 10:
                     self.reduce_population_size()
                 else:
                     self.restart(func)

            self.best_fitness_history.append(np.min(self.fitness))
            if self.budget <= 0:
                break

        return self.f_opt, self.x_opt