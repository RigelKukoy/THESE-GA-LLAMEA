import numpy as np

class AdaptiveAgingDE:
    def __init__(self, budget=10000, dim=10, initial_pop_size=50, archive_size=10, pbest_rate=0.1, F=0.5, CR=0.7, aging_rate=0.05, age_limit=5):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = initial_pop_size
        self.pop_size = initial_pop_size  # Dynamic population size
        self.archive_size = archive_size
        self.pbest_rate = pbest_rate
        self.F = F
        self.CR = CR
        self.aging_rate = aging_rate  # Probability of removing an individual
        self.age_limit = age_limit  # Max age before removal
        self.population = None
        self.fitness = None
        self.ages = None  # Age of each individual
        self.best_index = None
        self.archive = None
        self.archive_fitness = None
        self.f_opt = np.inf
        self.x_opt = None
        self.success_history_F = []
        self.success_history_CR = []

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.ages = np.zeros(self.pop_size)
        self.budget -= self.pop_size
        self.best_index = np.argmin(self.fitness)
        self.f_opt = self.fitness[self.best_index]
        self.x_opt = self.population[self.best_index].copy()
        self.archive = np.zeros((self.archive_size, self.dim))
        self.archive_fitness = np.full(self.archive_size, np.inf)

    def current_to_pbest_mutation(self, x, pbest_indices, x_r1, F):
        x_pbest = self.population[np.random.choice(pbest_indices)]
        return x + F * (x_pbest - x) + F * (x_r1 - x)
    
    def update_archive(self, x, f_x):
        if np.any(f_x < self.archive_fitness):
            worst_index = np.argmax(self.archive_fitness)
            self.archive[worst_index] = x
            self.archive_fitness[worst_index] = f_x

    def remove_stagnant_individuals(self):
        # Identify stagnant individuals based on age
        stagnant_indices = np.where(self.ages >= self.age_limit)[0]
        num_to_remove = len(stagnant_indices)

        if num_to_remove > 0:
            # Replace stagnant individuals with new random individuals
            new_individuals = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(num_to_remove, self.dim))
            new_fitness = np.array([func(x) for x in new_individuals])
            self.budget -= num_to_remove

            self.population[stagnant_indices] = new_individuals
            self.fitness[stagnant_indices] = new_fitness
            self.ages[stagnant_indices] = 0  # Reset age

    def __call__(self, func):
        self.initialize_population(func)

        while self.budget > 0:
            self.remove_stagnant_individuals()

            for i in range(self.pop_size):
                # Parameter adaptation using success history
                if self.success_history_F:
                    self.F = np.random.choice(self.success_history_F)
                if self.success_history_CR:
                    self.CR = np.random.choice(self.success_history_CR)
                    
                # Mutation
                pbest_count = max(1, int(self.pbest_rate * self.pop_size))
                pbest_indices = np.argsort(self.fitness)[:pbest_count]
                
                indices = np.random.choice(self.pop_size, 1, replace=False)
                x_r1 = self.population[indices[0]]
                
                v = self.current_to_pbest_mutation(self.population[i], pbest_indices, x_r1, self.F)
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
                    self.success_history_F.append(self.F)
                    self.success_history_CR.append(self.CR)
                    
                    self.fitness[i] = f_u
                    self.population[i] = u
                    self.ages[i] = 0  # Reset age if improved
                    
                    if f_u < self.f_opt:
                        self.f_opt = f_u
                        self.x_opt = u.copy()
                        
                    self.update_archive(u, f_u)
                else:
                    # Increment age if not improved
                    self.ages[i] += 1

                # Limit the size of the success history
                self.success_history_F = self.success_history_F[-10:]
                self.success_history_CR = self.success_history_CR[-10:]
            
            #Probabilistically remove individuals from the population
            removal_prob = self.aging_rate
            remove = np.random.rand(self.pop_size) < removal_prob
            num_removed = np.sum(remove)

            if num_removed > 0:

                new_individuals = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(num_removed, self.dim))
                new_fitness = np.array([func(x) for x in new_individuals])
                self.budget -= num_removed

                self.population[remove] = new_individuals
                self.fitness[remove] = new_fitness
                self.ages[remove] = 0  # Reset age

        return self.f_opt, self.x_opt