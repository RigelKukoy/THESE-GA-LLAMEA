import numpy as np

class ShiftingMomentumAdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=40, Cr=0.9, F_initial=0.5, stagnation_threshold=100, restart_prob=0.1, curvature_window=50, momentum=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.Cr = Cr
        self.F = F_initial
        self.stagnation_threshold = stagnation_threshold
        self.restart_prob = restart_prob
        self.curvature_window = curvature_window
        self.momentum = momentum
        self.best_fitness_history = []
        self.last_improvement = 0
        self.generation = 0
        self.fitness_trend = [] # Store recent fitness values to estimate curvature

    def __call__(self, func):
        # Initialization
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size
        
        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]
        self.best_fitness_history.append(self.f_opt)
        self.fitness_trend.append(self.f_opt)
        self.last_improvement = 0
        self.generation = 0
        self.previous_mutation = np.zeros((self.pop_size, self.dim)) # Initialize previous mutation direction

        while self.budget > self.pop_size:
            # Mutation and Crossover
            new_population = np.copy(population)
            for i in range(self.pop_size):
                # Shifting Ring Topology
                shift = np.random.randint(1, self.pop_size // 4 + 1)  # Shift by a random amount
                idx_prev = (i - shift) % self.pop_size
                idx_next = (i + shift) % self.pop_size
                
                # Momentum-based Mutation
                r1, r2 = np.random.choice(self.pop_size, 2, replace=False)
                
                # Calculate mutation vector with momentum
                mutation_vector = self.F * (population[r1] - population[r2])
                mutation_vector = self.momentum * self.previous_mutation[i] + (1 - self.momentum) * mutation_vector
                
                mutant = population[i] + mutation_vector
                
                # Store the mutation vector for the next iteration
                self.previous_mutation[i] = mutation_vector
                
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
            self.fitness_trend.append(self.f_opt)
            if len(self.fitness_trend) > self.curvature_window:
                self.fitness_trend.pop(0)
            
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

            # Adaptive Parameter Control based on Curvature
            if len(self.fitness_trend) >= self.curvature_window:
                # Estimate curvature (simplified as the difference between the first and last fitness values)
                curvature = self.fitness_trend[-1] - self.fitness_trend[0]

                # Adjust Cr and F based on curvature
                if curvature > 0:  # Positive curvature indicates slow progress
                    self.Cr *= 0.9  # Reduce crossover rate to promote exploration
                    self.F *= 1.1   # Increase mutation rate to escape local optima
                else:  # Negative curvature indicates good progress
                    self.Cr *= 1.1  # Increase crossover rate to promote exploitation
                    self.F *= 0.9   # Reduce mutation rate to refine the solution
                
                self.Cr = np.clip(self.Cr, 0.1, 0.95)
                self.F = np.clip(self.F, 0.1, 2.0) #prevent F from getting too high.

            self.generation += 1
        
        return self.f_opt, self.x_opt