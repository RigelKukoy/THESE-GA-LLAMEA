import numpy as np

class DynamicPopulationDE:
    def __init__(self, budget=10000, dim=10, initial_pop_size=50, Cr=0.9, F=0.5, pop_size_reduction_factor=0.9, pop_size_increase_trigger=0.05):
        self.budget = budget
        self.dim = dim
        self.pop_size = initial_pop_size
        self.Cr = Cr
        self.F = F
        self.pop_size_reduction_factor = pop_size_reduction_factor
        self.pop_size_increase_trigger = pop_size_increase_trigger
        self.population = None
        self.fitness = None
        self.f_opt = np.Inf
        self.x_opt = None
        self.generation = 0

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        best_index = np.argmin(self.fitness)
        self.f_opt = self.fitness[best_index]
        self.x_opt = self.population[best_index]

    def adjust_population_size(self):
        # Calculate population diversity (average distance to centroid)
        centroid = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - centroid, axis=1)
        avg_distance = np.mean(distances)

        # Reduce population size if diversity is low
        if avg_distance < self.pop_size_increase_trigger: # self.dim * 0.1
            new_pop_size = int(self.pop_size * self.pop_size_reduction_factor)
            if new_pop_size > 10: # Minimum population size
                self.pop_size = new_pop_size
                # Select the best individuals to keep
                indices = np.argsort(self.fitness)[:self.pop_size]
                self.population = self.population[indices]
                self.fitness = self.fitness[indices]
                print(f"Reduced population size to {self.pop_size}")
        elif self.generation % 50 == 0 and self.pop_size < 100: #Increase population slowly if not already too high
            self.pop_size = min(self.pop_size+5, 100)
            print(f"Increasing population size to {self.pop_size}")
            

    def distance_based_mutation(self, i):
         # Select two random indices, ensuring they are different from i
        idxs = np.random.choice(self.population.shape[0], 2, replace=False)
        while i in idxs:
            idxs = np.random.choice(self.population.shape[0], 2, replace=False)
        
        x_r1, x_r2 = self.population[idxs[0]], self.population[idxs[1]]
        
        # Calculate the distance between the individual and the best individual
        distance = np.linalg.norm(self.population[i] - self.x_opt)
        
        # Scale the mutation factor based on the distance
        scaled_F = self.F * (1 + distance)
        
        # Create the mutant vector
        mutant = self.population[i] + scaled_F * (x_r1 - x_r2)
        mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)
        return mutant

    def __call__(self, func):
        self.initialize_population(func)

        while self.budget > 0:
            new_population = np.copy(self.population)
            new_fitness = np.copy(self.fitness)

            for i in range(self.pop_size):
                # Mutation
                mutant = self.distance_based_mutation(i)
                
                # Crossover
                for j in range(self.dim):
                    if np.random.rand() < self.Cr:
                        new_population[i, j] = mutant[j]
                    else:
                        new_population[i, j] = self.population[i, j]

                new_population[i] = np.clip(new_population[i], func.bounds.lb, func.bounds.ub)

                # Evaluation
                new_fitness_i = func(new_population[i])
                self.budget -= 1

                if new_fitness_i < self.fitness[i]:
                    self.population[i] = new_population[i]
                    self.fitness[i] = new_fitness_i

                    if new_fitness_i < self.f_opt:
                        self.f_opt = new_fitness_i
                        self.x_opt = new_population[i]
            
            self.generation += 1
            if self.generation % 10 == 0:
                self.adjust_population_size()


        return self.f_opt, self.x_opt