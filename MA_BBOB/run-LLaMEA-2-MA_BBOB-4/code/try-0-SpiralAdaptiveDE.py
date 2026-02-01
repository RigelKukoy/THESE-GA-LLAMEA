import numpy as np

class SpiralAdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=40, Cr=0.9, F_initial=0.5, stagnation_threshold=100, restart_prob=0.1, spiral_factor=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.Cr = Cr
        self.F = F_initial
        self.stagnation_threshold = stagnation_threshold
        self.restart_prob = restart_prob
        self.spiral_factor = spiral_factor
        self.best_fitness_history = []
        self.last_improvement = 0
        self.generation = 0
        self.exploration_rate = 0.7  # Initial exploration rate

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
                if np.random.rand() < self.exploration_rate:
                    # Exploration: Spiral Dynamic inspired mutation
                    r1, r2 = np.random.choice(self.pop_size, 2, replace=False)
                    center = self.x_opt  # Spiral towards the current best
                    radius = np.linalg.norm(population[i] - center)
                    angle = np.random.uniform(0, 2 * np.pi)
                    mutant = center + radius * np.exp(self.spiral_factor * angle) * np.array([np.cos(angle), np.sin(angle)])[:self.dim]

                else:
                    # Exploitation: Standard DE mutation
                    r1, r2, r3 = np.random.choice(self.pop_size, 3, replace=False)
                    mutant = population[r1] + self.F * (population[r2] - population[r3])

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
                    
            # Dynamically adjust exploration rate
            if self.generation % 50 == 0:
                if (self.generation - self.last_improvement) < self.stagnation_threshold // 2:
                     self.exploration_rate = min(1.0, self.exploration_rate + 0.05)  # Increase exploration
                else:
                     self.exploration_rate = max(0.1, self.exploration_rate - 0.05)  # Decrease exploration
                    
            self.generation += 1
        
        return self.f_opt, self.x_opt