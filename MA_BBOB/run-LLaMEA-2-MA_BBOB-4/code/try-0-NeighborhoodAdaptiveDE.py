import numpy as np

class NeighborhoodAdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=40, Cr=0.9, F_initial=0.5, neighborhood_size=5, diversity_threshold=0.1, fitness_scaling=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.Cr = Cr
        self.F = F_initial
        self.neighborhood_size = neighborhood_size
        self.diversity_threshold = diversity_threshold
        self.fitness_scaling = fitness_scaling
        self.generation = 0

    def __call__(self, func):
        # Initialization
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size
        
        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]
        
        while self.budget > self.pop_size:
            new_population = np.copy(population)
            new_fitness = np.zeros(self.pop_size)

            for i in range(self.pop_size):
                # Neighborhood Selection
                neighborhood_indices = np.random.choice(self.pop_size, self.neighborhood_size, replace=False)
                neighborhood = population[neighborhood_indices]
                neighborhood_fitness = fitness[neighborhood_indices]
                
                # Fitness Landscape Analysis (Neighborhood Diversity)
                diversity = np.std(neighborhood)
                
                # Adaptive F
                if diversity < self.diversity_threshold:
                    F = self.F * 1.5  # Increase mutation strength if neighborhood is too homogeneous
                else:
                    F = self.F * 0.8  # Decrease mutation strength if neighborhood is diverse
                F = np.clip(F, 0.1, 2.0)
                
                # Mutation
                r1, r2, r3 = np.random.choice(self.pop_size, 3, replace=False)
                mutant = population[i] + F * (population[r1] - population[r2])
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)
                
                # Crossover
                for j in range(self.dim):
                    if np.random.rand() < self.Cr:
                        new_population[i, j] = mutant[j]
                    # else: new_population[i,j] = population[i,j] #no change

                new_population[i] = np.clip(new_population[i], func.bounds.lb, func.bounds.ub)
                
                # Evaluation
                new_fitness[i] = func(new_population[i])
                self.budget -= 1

            # Selection with Fitness-Dependent Reproduction Probability
            for i in range(self.pop_size):
                # Scale fitness values to create probabilities
                scaled_fitness = np.exp(-self.fitness_scaling * (new_fitness - np.min(new_fitness)))
                probabilities = scaled_fitness / np.sum(scaled_fitness)

                # Randomly choose whether to replace based on the fitness-dependent probabilities
                if np.random.rand() < probabilities[i]: # Probability is proportional to fitness
                  if new_fitness[i] < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]

                    if new_fitness[i] < self.f_opt:
                        self.f_opt = new_fitness[i]
                        self.x_opt = new_population[i]

            self.generation += 1

        return self.f_opt, self.x_opt