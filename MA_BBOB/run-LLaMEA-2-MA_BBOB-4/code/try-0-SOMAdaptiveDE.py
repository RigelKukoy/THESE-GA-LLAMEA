import numpy as np
from minisom import MiniSom

class SOMAdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=40, som_grid_size=5, F_init=0.5, Cr_init=0.7, stagnation_threshold=50):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.som_grid_size = som_grid_size
        self.F = F_init
        self.Cr = Cr_init
        self.stagnation_threshold = stagnation_threshold
        self.best_fitness_history = []
        self.stagnation_counter = 0
        self.som = MiniSom(som_grid_size, som_grid_size, dim, sigma=0.3, learning_rate=0.5)
        self.last_improvement = 0

    def __call__(self, func):
        # Initialization
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size
        
        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)].copy()
        self.best_fitness_history.append(self.f_opt)
        
        generation = 0

        while self.budget > self.pop_size:
            # Train SOM
            self.som.train_random(population, 10)

            # Assign individuals to SOM nodes
            node_assignments = [self.som.winner(x) for x in population]

            # Calculate cluster performance and diversity
            cluster_fitnesses = {}
            cluster_diversities = {}
            for i in range(self.som_grid_size):
                for j in range(self.som_grid_size):
                    cluster_individuals = [k for k, assignment in enumerate(node_assignments) if assignment == (i, j)]
                    if cluster_individuals:
                        cluster_fitnesses[(i, j)] = np.mean(fitness[cluster_individuals])
                        cluster_diversities[(i, j)] = np.std(population[cluster_individuals])

            # Adaptive Parameter Adjustment based on SOM clusters
            new_population = np.copy(population)
            for i in range(self.pop_size):
                cluster = node_assignments[i]
                if cluster in cluster_fitnesses:
                    # Adjust F and Cr based on cluster performance
                    if cluster_fitnesses[cluster] < np.mean(fitness): #Good cluster
                        self.F = max(0.1, self.F * 0.95)  #Exploitation
                        self.Cr = min(0.9, self.Cr * 1.05)
                    else: # Bad cluster
                        self.F = min(1.0, self.F * 1.05) # Exploration
                        self.Cr = max(0.1, self.Cr * 0.95)
                    
                    # Adjust F and Cr based on cluster diversity
                    if cluster_diversities.get(cluster,0) > np.mean(np.std(population, axis=0)):
                         self.F = min(1.0, self.F * 1.02) # Exploration
                         self.Cr = max(0.1, self.Cr * 0.98)
                    else:
                         self.F = max(0.1, self.F * 0.98)  #Exploitation
                         self.Cr = min(0.9, self.Cr * 1.02)

            # Mutation and Crossover
                indices = [j for j in range(self.pop_size) if j != i]
                idxs = np.random.choice(indices, size=3, replace=False)
                x_r1, x_r2, x_r3 = population[idxs[0]], population[idxs[1]], population[idxs[2]]
                
                mutant = population[i] + self.F * (x_r1 - x_r2) + self.F * (x_r3 - population[i])
                
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
                        self.x_opt = new_population[i].copy()
                        self.last_improvement = generation
                        
            self.best_fitness_history.append(self.f_opt)

            #Stagnation check
            if generation > 0:
                if self.f_opt == self.best_fitness_history[-1]:
                    self.stagnation_counter += 1
                else:
                    self.stagnation_counter = 0
            
            if self.stagnation_counter > self.stagnation_threshold:
                # Reset population if stagnated
                population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
                fitness = np.array([func(x) for x in population])
                self.budget -= self.pop_size  # Adjust budget after re-evaluation
                self.stagnation_counter = 0
            
            generation += 1

        return self.f_opt, self.x_opt