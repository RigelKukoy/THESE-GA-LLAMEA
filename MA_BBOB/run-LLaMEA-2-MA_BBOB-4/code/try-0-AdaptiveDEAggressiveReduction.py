import numpy as np

class AdaptiveDEAggressiveReduction:
    def __init__(self, budget=10000, dim=10, pop_size=40, F_initial=0.5, Cr=0.9, stagnation_threshold=50, pop_size_reduction_factor=0.7, min_pop_size=5):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F_initial = F_initial
        self.Cr = Cr
        self.stagnation_threshold = stagnation_threshold
        self.pop_size_reduction_factor = pop_size_reduction_factor
        self.min_pop_size = min_pop_size
        self.best_fitness_history = []
        self.last_improvement = 0
        self.F = self.F_initial  # Initialize F

    def __call__(self, func):
        # Initialization
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size
        
        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]
        self.best_fitness_history.append(self.f_opt)
        
        generation = 0

        while self.budget > self.min_pop_size:  # Ensure budget is sufficient for min_pop_size evaluations
            # Mutation and Crossover
            new_population = np.copy(population)
            for i in range(self.pop_size):
                # Simplified Mutation: Adapt F based on recent success. No separate F selection.
                if generation > 0 and len(self.best_fitness_history) > 1:
                    if self.best_fitness_history[-1] < self.best_fitness_history[-2]:  # Improvement
                        self.F = max(0.1, self.F * 0.95)  # Reduce F if improving
                    else:
                        self.F = min(0.9, self.F * 1.05) #Increase F if stagnating

                indices = [j for j in range(self.pop_size) if j != i]
                idxs = np.random.choice(indices, size=3, replace=False)
                x_r1, x_r2, x_r3 = population[idxs[0]], population[idxs[1]], population[idxs[2]]
                
                mutant = population[i] + self.F * (x_r1 - x_r2) + self.F * (x_r3 - population[i])
                
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
            
            # Stagnation check and aggressive population size reduction
            if (generation - self.last_improvement) > self.stagnation_threshold:
                #Aggressive Population Size Reduction:
                self.pop_size = max(int(self.pop_size * self.pop_size_reduction_factor), self.min_pop_size)
                population = population[:self.pop_size]
                fitness = fitness[:self.pop_size]
                #print(f"Population reduced to {self.pop_size}")
                self.last_improvement = generation # Reset last improvement to avoid repeated reduction

            generation += 1
        
        return self.f_opt, self.x_opt