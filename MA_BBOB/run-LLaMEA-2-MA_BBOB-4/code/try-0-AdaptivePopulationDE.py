import numpy as np

class AdaptivePopulationDE:
    def __init__(self, budget=10000, dim=10, pop_size_initial=40, pop_size_min=10, pop_size_max=100, F=0.5, Cr=0.9, stagnation_threshold=100, local_search_probability=0.1, local_search_radius=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size_initial = pop_size_initial
        self.pop_size = pop_size_initial
        self.pop_size_min = pop_size_min
        self.pop_size_max = pop_size_max
        self.F = F
        self.Cr = Cr
        self.stagnation_threshold = stagnation_threshold
        self.local_search_probability = local_search_probability
        self.local_search_radius = local_search_radius
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
        
        while self.budget > self.pop_size_min:
            # Mutation and Crossover
            new_population = np.copy(population)
            for i in range(self.pop_size):
                indices = [j for j in range(self.pop_size) if j != i]
                idxs = np.random.choice(indices, size=3, replace=False)
                x_r1, x_r2, x_r3 = population[idxs[0]], population[idxs[1]], population[idxs[2]]
                
                mutant = population[i] + self.F * (x_r1 - x_r2)
                
                # Crossover
                for j in range(self.dim):
                    if np.random.rand() < self.Cr:
                        new_population[i, j] = mutant[j]
                    
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

            # Stagnation check and Local Search
            if (self.generation - self.last_improvement) > self.stagnation_threshold:
                if np.random.rand() < self.local_search_probability:
                    # Perform local search around the best solution
                    x_local = self.x_opt + np.random.uniform(-self.local_search_radius, self.local_search_radius, size=self.dim)
                    x_local = np.clip(x_local, func.bounds.lb, func.bounds.ub)
                    f_local = func(x_local)
                    self.budget -= 1

                    if f_local < self.f_opt:
                        self.f_opt = f_local
                        self.x_opt = x_local
                        self.last_improvement = self.generation
                        print("Local search improved the solution")
            
            # Population size adaptation
            if self.f_opt == self.best_fitness_history[-1] and len(self.best_fitness_history) > 1:  #Stagnation
                self.pop_size = max(self.pop_size_min, int(self.pop_size * 0.9)) #Reduce the population size
            else:
                 self.pop_size = min(self.pop_size_max, int(self.pop_size * 1.1)) #Increase the population size

            #Ensure pop_size never goes below the minimal population size.
            self.pop_size = max(self.pop_size, self.pop_size_min)
            #Regenerate the population:
            population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
            fitness = np.array([func(x) for x in population])
            self.budget -= self.pop_size
            self.f_opt = np.min(fitness)
            self.x_opt = population[np.argmin(fitness)]
            self.last_improvement = self.generation 
            
            self.generation += 1

        return self.f_opt, self.x_opt