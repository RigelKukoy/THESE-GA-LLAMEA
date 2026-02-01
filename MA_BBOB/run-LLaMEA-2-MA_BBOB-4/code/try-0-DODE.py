import numpy as np

class DODE:
    def __init__(self, budget=10000, dim=10, initial_pop_size=50, min_pop_size=10, max_pop_size=100, restart_factor=0.1):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = initial_pop_size
        self.min_pop_size = min_pop_size
        self.max_pop_size = max_pop_size
        self.restart_factor = restart_factor
        self.pop_size = initial_pop_size
        self.F = 0.5
        self.Cr = 0.9

    def __call__(self, func):
        # Initialization
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size
        
        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]
        
        iteration = 0
        no_improvement_count = 0
        
        while self.budget > self.min_pop_size:
            iteration += 1
            
            # Mutation and Crossover
            new_population = np.copy(population)
            new_fitness = np.zeros(self.pop_size)
            for i in range(self.pop_size):
                # Mutation
                indices = [j for j in range(self.pop_size) if j != i]
                idxs = np.random.choice(indices, size=3, replace=False)
                x_r1, x_r2, x_r3 = population[idxs[0]], population[idxs[1]], population[idxs[2]]
                mutant = population[i] + self.F * (x_r1 - x_r2) + self.F * (x_r3 - population[i])
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)
                
                # Crossover
                for j in range(self.dim):
                    if np.random.rand() < self.Cr:
                        new_population[i, j] = mutant[j]
                        
                new_population[i] = np.clip(new_population[i], func.bounds.lb, func.bounds.ub)
                new_fitness[i] = func(new_population[i])
                self.budget -= 1
                
            # Selection
            for i in range(self.pop_size):
                if new_fitness[i] < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]
                    
                    if new_fitness[i] < self.f_opt:
                        self.f_opt = new_fitness[i]
                        self.x_opt = new_population[i]
                        no_improvement_count = 0  # Reset counter
                else:
                    no_improvement_count += 1
            
            # Dynamic Population Size Adjustment
            if no_improvement_count > 50:
                self.pop_size = max(self.min_pop_size, int(self.pop_size * 0.9)) # Reduce population size if no improvement
            elif len(np.unique(fitness)) > self.pop_size * 0.8:
                self.pop_size = min(self.max_pop_size, int(self.pop_size * 1.1)) # Increase population size if diversity is high
                
            self.pop_size = int(np.clip(self.pop_size, self.min_pop_size, self.max_pop_size))
            
            # Orthogonal Learning (OL) - Applied probabilistically
            if np.random.rand() < 0.1:
                best_index = np.argmin(fitness)
                x_best = population[best_index].copy()
                
                # Generate orthogonal design points around the best solution
                orthogonal_points = self.generate_orthogonal_design(x_best, func.bounds.lb, func.bounds.ub, num_points=5)
                
                for x_ol in orthogonal_points:
                    f_ol = func(x_ol)
                    self.budget -= 1
                    if f_ol < self.f_opt:
                        self.f_opt = f_ol
                        self.x_opt = x_ol
                        
            # Restart mechanism
            if no_improvement_count > 200:  # If still no improvement after many iterations, restart
                population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
                fitness = np.array([func(x) for x in population])
                self.budget -= self.pop_size
                self.f_opt = np.min(fitness)
                self.x_opt = population[np.argmin(fitness)]
                no_improvement_count = 0
        return self.f_opt, self.x_opt

    def generate_orthogonal_design(self, x_center, lb, ub, num_points=5):
        design = []
        for _ in range(num_points):
            x = x_center + np.random.uniform(-0.1, 0.1, self.dim) * (ub - lb)  # Small perturbation
            x = np.clip(x, lb, ub)
            design.append(x)
        return np.array(design)