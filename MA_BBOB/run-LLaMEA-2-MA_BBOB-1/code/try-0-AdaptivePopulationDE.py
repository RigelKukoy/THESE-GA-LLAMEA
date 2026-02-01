import numpy as np

class AdaptivePopulationDE:
    def __init__(self, budget=10000, dim=10, initial_pop_size=50, F=0.5, CR=0.7, diversity_threshold=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = initial_pop_size
        self.F = F
        self.CR = CR
        self.diversity_threshold = diversity_threshold
        self.population = None
        self.fitness = None
        self.best_index = None
        self.f_opt = np.inf
        self.x_opt = None
        self.age = None

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.age = np.zeros(self.pop_size)
        self.budget -= self.pop_size
        self.best_index = np.argmin(self.fitness)
        self.f_opt = self.fitness[self.best_index]
        self.x_opt = self.population[self.best_index].copy()
        

    def adjust_population_size(self):
        """Adjust population size based on diversity and stagnation."""
        diversity = np.mean(np.linalg.norm(self.population - self.x_opt, axis=1)) / (np.linalg.norm(self.population[0] - self.population[1]) + 1e-8)
        
        if diversity < self.diversity_threshold and self.pop_size < 200:
            # Increase population size if diversity is low and population is not too large
            increase_amount = min(10, self.budget // 2)  # Limit increase amount and budget usage
            if increase_amount > 0:
                new_individuals = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(increase_amount, self.dim))
                new_fitness = np.array([func(x) for x in new_individuals])
                self.budget -= increase_amount

                self.population = np.vstack((self.population, new_individuals))
                self.fitness = np.concatenate((self.fitness, new_fitness))
                self.age = np.concatenate((self.age, np.zeros(increase_amount)))
                self.pop_size = len(self.population)
                self.best_index = np.argmin(self.fitness) #update
                self.f_opt = self.fitness[self.best_index]
                self.x_opt = self.population[self.best_index].copy()
        elif np.all(self.age > 50) and self.pop_size > 20:
            # Reduce population size if the population has stagnated and population is not too small
            reduction_amount = min(10, self.pop_size // 2) #limit reduction amount
            indices_to_remove = np.argsort(self.fitness)[-reduction_amount:] # Remove worst individuals
            self.population = np.delete(self.population, indices_to_remove, axis=0)
            self.fitness = np.delete(self.fitness, indices_to_remove)
            self.age = np.delete(self.age, indices_to_remove)
            self.pop_size = len(self.population)
            self.best_index = np.argmin(self.fitness) #update
            self.f_opt = self.fitness[self.best_index]
            self.x_opt = self.population[self.best_index].copy()
        
        self.age +=1  # Increment age for all individuals

    def __call__(self, func):
        self.initialize_population(func)

        while self.budget > 0:
            self.adjust_population_size()

            for i in range(self.pop_size):
                # Differential Evolution
                indices = np.random.choice(self.pop_size, 3, replace=False)
                x_r1, x_r2, x_r3 = self.population[indices]

                v = self.population[i] + self.F * (x_r1 - x_r2)

                v = np.clip(v, func.bounds.lb, func.bounds.ub)

                # Crossover
                j_rand = np.random.randint(self.dim)
                u = self.population[i].copy()
                for k in range(self.dim):
                    if np.random.rand() < self.CR or k == j_rand:
                        u[k] = v[k]

                f_u = func(u)
                self.budget -= 1

                # Selection
                if f_u < self.fitness[i]:
                    self.fitness[i] = f_u
                    self.population[i] = u
                    self.age[i] = 0  # Reset age if improved

                    if f_u < self.f_opt:
                        self.f_opt = f_u
                        self.x_opt = u.copy()
                        self.best_index = i


        return self.f_opt, self.x_opt