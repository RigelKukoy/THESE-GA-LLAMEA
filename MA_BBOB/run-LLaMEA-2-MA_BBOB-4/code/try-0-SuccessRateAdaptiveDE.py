import numpy as np

class SuccessRateAdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=40, Cr_initial=0.5, F_initial=0.7, success_rate_window=20, orthogonal_learning_prob=0.05):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.Cr = Cr_initial
        self.F = F_initial
        self.success_rate_window = success_rate_window
        self.orthogonal_learning_prob = orthogonal_learning_prob
        self.success_rates_Cr = []
        self.success_rates_F = []
        self.archive = []
        self.archive_fitness = []

    def __call__(self, func):
        # Initialization
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size
        
        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]

        while self.budget > self.pop_size:
            # Mutation and Crossover
            new_population = np.copy(population)
            successful_Cr = []
            successful_F = []

            for i in range(self.pop_size):
                # Mutation
                r1, r2, r3 = np.random.choice(self.pop_size, 3, replace=False)
                mutant = population[r1] + self.F * (population[r2] - population[r3])
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

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
                    successful_Cr.append(self.Cr)
                    successful_F.append(self.F)
                    
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]
                    
                    if new_fitness[i] < self.f_opt:
                        self.f_opt = new_fitness[i]
                        self.x_opt = new_population[i]
                        
                        #Archive successful solutions
                        if len(self.archive) < self.pop_size:
                           self.archive.append(new_population[i])
                           self.archive_fitness.append(new_fitness[i])
                        else:
                            max_archive_fitness_index = np.argmax(self.archive_fitness)
                            if new_fitness[i] < self.archive_fitness[max_archive_fitness_index]:
                                self.archive[max_archive_fitness_index] = new_population[i]
                                self.archive_fitness[max_archive_fitness_index] = new_fitness[i]
                            
            # Adaptive Parameter Control based on Success Rate
            if successful_Cr:
                self.success_rates_Cr.append(np.mean(successful_Cr))
            else:
                self.success_rates_Cr.append(0)  # No successful Cr value

            if successful_F:
                self.success_rates_F.append(np.mean(successful_F))
            else:
                self.success_rates_F.append(0)

            if len(self.success_rates_Cr) > self.success_rate_window:
                self.success_rates_Cr.pop(0)
            if len(self.success_rates_F) > self.success_rate_window:
                self.success_rates_F.pop(0)

            # Adjust Cr and F based on average success rate over the window
            if self.success_rates_Cr:
                avg_success_Cr = np.mean(self.success_rates_Cr)
                self.Cr = min(0.95, max(0.1, avg_success_Cr * 1.2))  # Adjust Cr, prevent too high or low
            if self.success_rates_F:
                avg_success_F = np.mean(self.success_rates_F)
                self.F = min(1.2, max(0.2, avg_success_F * 1.1)) # Adjust F

            # Orthogonal Learning
            if np.random.rand() < self.orthogonal_learning_prob and len(self.archive) >= 2:
                # Select two parents from the archive
                parent1_idx, parent2_idx = np.random.choice(len(self.archive), 2, replace=False)
                parent1 = self.archive[parent1_idx]
                parent2 = self.archive[parent2_idx]

                # Generate a new individual based on orthogonal array design
                orthogonal_individual = np.zeros(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < 0.5:
                        orthogonal_individual[j] = parent1[j]
                    else:
                        orthogonal_individual[j] = parent2[j]

                orthogonal_individual = np.clip(orthogonal_individual, func.bounds.lb, func.bounds.ub)
                orthogonal_fitness = func(orthogonal_individual)
                self.budget -= 1

                # Replace the worst individual in the population with the orthogonal individual if it's better
                worst_idx = np.argmax(fitness)
                if orthogonal_fitness < fitness[worst_idx]:
                    population[worst_idx] = orthogonal_individual
                    fitness[worst_idx] = orthogonal_fitness

                    if orthogonal_fitness < self.f_opt:
                        self.f_opt = orthogonal_fitness
                        self.x_opt = orthogonal_individual
        
        return self.f_opt, self.x_opt