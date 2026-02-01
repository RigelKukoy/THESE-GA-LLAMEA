import numpy as np

class RestartDE:
    def __init__(self, budget=10000, dim=10, pop_size=40, F=0.5, Cr_initial=0.9, stagnation_threshold=100, diversity_threshold=0.1, restart_probability=0.2):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.Cr_initial = Cr_initial
        self.Cr = Cr_initial  # Initialize Cr
        self.stagnation_threshold = stagnation_threshold
        self.diversity_threshold = diversity_threshold
        self.restart_probability = restart_probability
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
            
            # Adjust crossover rate based on diversity
            if diversity < self.diversity_threshold:
                self.Cr = min(1.0, self.Cr + 0.05)  # Increase Cr if diversity is low
            else:
                self.Cr = max(0.1, self.Cr - 0.025)  # Decrease Cr if diversity is high

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

            # Stagnation check and restart
            if (generation - self.last_improvement) > self.stagnation_threshold:
                if np.random.rand() < self.restart_probability:
                    # Restart: Reinitialize the population
                    population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
                    fitness = np.array([func(x) for x in population])
                    self.budget -= self.pop_size
                    self.f_opt = np.min(fitness)
                    self.x_opt = population[np.argmin(fitness)]
                    self.last_improvement = generation  # Reset last improvement
                    self.Cr = self.Cr_initial # Reset Cr
                    print("Restarting population")

            generation += 1

        return self.f_opt, self.x_opt