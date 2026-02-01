import numpy as np

class AdaptiveDynamicDE:
    def __init__(self, budget=10000, dim=10, pop_size_min=20, pop_size_max=100, archive_size=50, F_min=0.1, F_max=1.0, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size_min = pop_size_min
        self.pop_size_max = pop_size_max
        self.pop_size = pop_size_max  # Initialize with the maximum population size
        self.archive_size = archive_size
        self.F_min = F_min
        self.F_max = F_max
        self.CR = CR
        self.population = None
        self.fitness = None
        self.archive = None
        self.best_fitness_history = []

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        self.archive = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.archive_size, self.dim))
        
    def adaptive_mutation_factor(self):
        # Adjust F based on the recent history of fitness improvements.
        if len(self.best_fitness_history) < 5:
            return np.random.uniform(self.F_min, self.F_max) # Return a random value initially
        
        recent_improvements = [self.best_fitness_history[i] - self.best_fitness_history[i-1] for i in range(1, len(self.best_fitness_history))]
        avg_improvement = np.mean(recent_improvements)
        
        # If improvement is high, use a smaller F for exploitation, else a larger F for exploration.
        if avg_improvement > 0: # Higher value indicates decreasing fitness
            F = self.F_min + (self.F_max - self.F_min) * np.exp(-avg_improvement * 10) # Exponential decay
        else:
            F = self.F_max
            
        return np.clip(F, self.F_min, self.F_max)

    def mutate(self, x_i, func):
        indices = np.random.choice(self.pop_size + self.archive_size, size=3, replace=False)
        
        if indices[0] < self.pop_size:
            x_r1 = self.population[indices[0]]
        else:
            x_r1 = self.archive[indices[0] - self.pop_size]

        if indices[1] < self.pop_size:
            x_r2 = self.population[indices[1]]
        else:
            x_r2 = self.archive[indices[1] - self.pop_size]
            
        if indices[2] < self.pop_size:
            x_r3 = self.population[indices[2]]
        else:
            x_r3 = self.archive[indices[2] - self.pop_size]

        F = self.adaptive_mutation_factor()
        return x_r1 + F * (x_r2 - x_r3)

    def crossover(self, x_i, v_i):
        u_i = np.copy(x_i)
        j_rand = np.random.randint(self.dim)
        for j in range(self.dim):
            if np.random.rand() <= self.CR or j == j_rand:
                u_i[j] = v_i[j]
        return u_i

    def repair(self, x, func):
         return np.clip(x, func.bounds.lb, func.bounds.ub)

    def update_archive(self):
        # Randomly replace archive members with population members
        indices = np.random.choice(self.archive_size, size=self.pop_size, replace=False)
        self.archive[indices] = self.population

    def adjust_population_size(self):
        # Dynamically adjust population size based on stagnation.
        if len(self.best_fitness_history) > 10:
            recent_improvements = [self.best_fitness_history[i] - self.best_fitness_history[i-1] for i in range(1, len(self.best_fitness_history))]
            avg_improvement = np.mean(recent_improvements)
            
            if avg_improvement > -1e-6: # Stagnation detected (little or no recent improvement)
                self.pop_size = max(self.pop_size_min, int(self.pop_size * 0.9))  # Reduce population
            else:
                self.pop_size = min(self.pop_size_max, int(self.pop_size * 1.1)) # Increase population, but no more than the maximum
                
            self.pop_size = int(self.pop_size)

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        self.initialize_population(func)
        self.best_fitness_history.append(np.min(self.fitness))

        iteration = 0
        while self.budget > self.pop_size_min: #Ensure budget is large enough to prevent errors
            iteration += 1

            #Dynamic population adjustment
            self.adjust_population_size()

            # Resample if pop size changed
            if self.pop_size != len(self.population):
                if self.pop_size < len(self.population):
                    indices = np.argsort(self.fitness)[:self.pop_size]  # Keep best
                    self.population = self.population[indices]
                    self.fitness = self.fitness[indices]
                else:  #self.pop_size > len(self.population)
                    num_new = self.pop_size - len(self.population)
                    new_individuals = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(num_new, self.dim))
                    new_fitness = np.array([func(x) for x in new_individuals])
                    self.budget -= num_new

                    self.population = np.vstack((self.population, new_individuals))
                    self.fitness = np.concatenate((self.fitness, new_fitness))

            for i in range(self.pop_size):
                # Mutation
                v_i = self.mutate(self.population[i], func)
                
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

            self.update_archive()
            self.best_fitness_history.append(np.min(self.fitness))


        return self.f_opt, self.x_opt