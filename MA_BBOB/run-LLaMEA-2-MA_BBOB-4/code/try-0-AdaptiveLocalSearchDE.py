import numpy as np

class AdaptiveLocalSearchDE:
    def __init__(self, budget=10000, dim=10, pop_size=40, F=0.5, Cr=0.9, stagnation_threshold=50, diversity_threshold=0.1, local_search_probability=0.1, local_search_radius=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.Cr = Cr
        self.stagnation_threshold = stagnation_threshold
        self.diversity_threshold = diversity_threshold
        self.local_search_probability = local_search_probability
        self.local_search_radius = local_search_radius
        self.best_fitness_history = []
        self.last_improvement = 0

    def __call__(self, func):
        # Initialization
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size
        
        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]
        self.best_fitness_history.append(self.f_opt)
        
        generation = 0

        while self.budget > self.pop_size:
            # Calculate population diversity (variance of each dimension)
            diversity = np.mean(np.var(population, axis=0))
            
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
                        self.last_improvement = generation
                        
            self.best_fitness_history.append(self.f_opt)

            # Adaptive Local Search
            if (generation - self.last_improvement) > self.stagnation_threshold and diversity < self.diversity_threshold:
                for i in range(self.pop_size):
                    if np.random.rand() < self.local_search_probability:
                        # Perform local search around individual i
                        x_current = population[i].copy()
                        f_current = fitness[i]
                        
                        for _ in range(5):  # Small budget for local search
                            x_new = x_current + np.random.uniform(-self.local_search_radius, self.local_search_radius, size=self.dim)
                            x_new = np.clip(x_new, func.bounds.lb, func.bounds.ub)
                            f_new = func(x_new)
                            self.budget -= 1
                            if self.budget <= 0:
                                return self.f_opt, self.x_opt
                                
                            if f_new < f_current:
                                x_current = x_new
                                f_current = f_new
                        
                        # Update population if local search finds a better solution
                        if f_current < fitness[i]:
                            population[i] = x_current
                            fitness[i] = f_current
                            if f_current < self.f_opt:
                                self.f_opt = f_current
                                self.x_opt = x_current
                                self.last_improvement = generation


            generation += 1

        return self.f_opt, self.x_opt