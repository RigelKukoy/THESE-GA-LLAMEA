import numpy as np

class VarianceAdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=40, Cr=0.9, F_initial=0.5, stagnation_threshold=100, restart_prob=0.1, local_search_prob=0.1, local_search_step_size=0.1, step_size_adaptation_rate=0.9, success_threshold=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.Cr = Cr
        self.F = F_initial
        self.stagnation_threshold = stagnation_threshold
        self.restart_prob = restart_prob
        self.local_search_prob = local_search_prob
        self.local_search_step_size = local_search_step_size
        self.step_size_adaptation_rate = step_size_adaptation_rate
        self.success_threshold = success_threshold
        self.best_fitness_history = []
        self.last_improvement = 0
        self.generation = 0

    def __call__(self, func):
        # Initialization
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size
        
        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]
        self.best_fitness_history.append(self.f_opt)
        self.last_improvement = 0
        self.generation = 0

        while self.budget > self.pop_size:
            # Mutation and Crossover
            new_population = np.copy(population)
            for i in range(self.pop_size):
                # Calculate variance of the fitness landscape around the individual
                neighborhood_size = min(self.pop_size // 5, 5)  #Adjust neighborhood size as needed
                neighbor_indices = np.random.choice(self.pop_size, size=neighborhood_size, replace=False)
                neighborhood_fitness = fitness[neighbor_indices]
                fitness_variance = np.var(neighborhood_fitness)

                # Modified Cauchy mutation: scale Cauchy noise by fitness variance
                cauchy_noise = self.F * np.random.standard_cauchy(size=self.dim) * (1 + fitness_variance) # Scale by variance
                mutant = population[i] + cauchy_noise

                # Crossover
                for j in range(self.dim):
                    if np.random.rand() < self.Cr:
                        new_population[i, j] = mutant[j]
                    else:
                        new_population[i, j] = population[i, j]
                
                new_population[i] = np.clip(new_population[i], func.bounds.lb, func.bounds.ub)
            
            # Evaluation
            new_fitness = np.array([func(x) for x in new_population])
            self.budget -= self.pop_size
            
            # Selection
            for i in range(self.pop_size):
                if new_fitness[i] < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]
                    
                    if new_fitness[i] < self.f_opt:
                        self.f_opt = new_fitness[i]
                        self.x_opt = new_population[i]
                        self.last_improvement = self.generation
                        
            self.best_fitness_history.append(self.f_opt)
            
            # Stagnation check and restart
            if (self.generation - self.last_improvement) > self.stagnation_threshold:
                if np.random.rand() < self.restart_prob:
                    # Restart the population
                    population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
                    fitness = np.array([func(x) for x in population])
                    self.budget -= self.pop_size
                    self.f_opt = np.min(fitness)
                    self.x_opt = population[np.argmin(fitness)]
                    self.last_improvement = self.generation
                    self.F = 0.5 #reset F
                else:
                    # Adaptive F: Reduce mutation strength upon stagnation
                    self.F *= 0.9  # Reduce F, but prevent it from becoming zero.
                    self.F = max(self.F, 0.1)
            
            # Local Search
            if np.random.rand() < self.local_search_prob:
                idx = np.random.randint(0, self.pop_size)
                current_fitness = fitness[idx]
                # Apply small perturbation to the selected individual
                new_individual = population[idx] + self.local_search_step_size * np.random.normal(0, 1, self.dim)
                new_individual = np.clip(new_individual, func.bounds.lb, func.bounds.ub)
                new_fitness_local = func(new_individual)
                self.budget -= 1

                if new_fitness_local < current_fitness:
                    population[idx] = new_individual
                    fitness[idx] = new_fitness_local
                    # Adapt step size: increase if successful
                    self.local_search_step_size /= self.step_size_adaptation_rate #increase step size
                    if fitness[idx] < self.f_opt:
                        self.f_opt = fitness[idx]
                        self.x_opt = population[idx]
                        self.last_improvement = self.generation

                else:
                     # Adapt step size: decrease if unsuccessful
                     self.local_search_step_size *= self.step_size_adaptation_rate #decrease step size

            self.local_search_step_size = np.clip(self.local_search_step_size, 0.0001, 0.5) #Bound step size
            self.generation += 1
        
        return self.f_opt, self.x_opt