import numpy as np
from sklearn.cluster import KMeans

class AdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, archive_size=50, orthogonal_learning_rate=0.1, restart_patience=50, archive_prob=0.1, current_to_best_prob=0.2, cluster_num = 5):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.archive_size = archive_size
        self.orthogonal_learning_rate = orthogonal_learning_rate
        self.restart_patience = restart_patience
        self.stagnation_counter = 0
        self.previous_best_fitness = np.Inf
        self.archive = []
        self.archive_fitnesses = []
        self.archive_F = []
        self.archive_CR = []
        self.archive_prob = archive_prob
        self.current_to_best_prob = current_to_best_prob
        self.cluster_num = cluster_num
        self.F = np.full(pop_size, 0.5)
        self.CR = np.full(pop_size, 0.7)

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        
        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size

        # Update optimal solution
        best_index = np.argmin(fitness)
        if fitness[best_index] < self.f_opt:
            self.f_opt = fitness[best_index]
            self.x_opt = population[best_index]
            self.stagnation_counter = 0
            self.previous_best_fitness = self.f_opt
        else:
            self.stagnation_counter +=1

        generation = 0
        while self.budget > 0:
            generation += 1
            new_population = np.copy(population)
            new_fitness = np.copy(fitness)

            # Clustering for diversity maintenance
            kmeans = KMeans(n_clusters=self.cluster_num, random_state=0, n_init = 'auto')
            clusters = kmeans.fit_predict(population)

            for i in range(self.pop_size):
                # Adaptive F and CR, individual-based
                F = self.F[i] + np.random.normal(0, 0.1)
                CR = self.CR[i] + np.random.normal(0, 0.1)
                F = np.clip(F, 0.1, 1.0)
                CR = np.clip(CR, 0.1, 1.0)

                # Mutation: Combining current-to-best with random mutation
                indices = np.random.choice(self.pop_size, 2, replace=False)
                x1, x2 = population[indices]
                
                if np.random.rand() < self.current_to_best_prob:
                    mutant = population[i] + F * (self.x_opt - population[i]) + F * (x1 - x2)
                else:
                    x3 = population[np.random.choice(self.pop_size)]
                    mutant = x1 + F * (x2 - x3)
                
                # Incorporate archive
                if len(self.archive) > 0 and np.random.rand() < self.archive_prob:
                    archive_index = np.random.randint(len(self.archive))
                    mutant = x1 + F * (x2 - self.archive[archive_index])

                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                # Crossover
                trial = np.copy(population[i])
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                
                # Orthogonal Learning
                if np.random.rand() < self.orthogonal_learning_rate:
                    orthogonal_vector = self.x_opt + np.random.normal(0, 0.1, self.dim)  # Perturb best solution
                    orthogonal_vector = np.clip(orthogonal_vector, func.bounds.lb, func.bounds.ub)
                    trial = 0.5 * (trial + orthogonal_vector)  # Combine with trial vector
                    trial = np.clip(trial, func.bounds.lb, func.bounds.ub)

                # Selection
                f_trial = func(trial)
                self.budget -= 1
                if f_trial < fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = f_trial

                    # Update F and CR values
                    self.F[i] = 0.9 * self.F[i] + 0.1 * F # Exponential smoothing
                    self.CR[i] = 0.9 * self.CR[i] + 0.1 * CR

                    # Add replaced vector to archive
                    if len(self.archive) < self.archive_size:
                        self.archive.append(population[i])
                    else:
                        self.archive[np.random.randint(self.archive_size)] = population[i]

                    # Update optimal solution
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                        self.stagnation_counter = 0
                        self.previous_best_fitness = self.f_opt
                else:
                    # If trial is worse, penalize F and CR
                    self.F[i] = np.clip(self.F[i] - 0.1, 0.1, 1.0)
                    self.CR[i] = np.clip(self.CR[i] - 0.1, 0.1, 1.0)

            population = new_population
            fitness = new_fitness
            
            # Restart population if stagnating using opposition-based learning
            if self.stagnation_counter > self.restart_patience:
                # Generate opposition population
                opposition_population = func.bounds.ub + func.bounds.lb - population
                opposition_fitness = np.array([func(x) for x in opposition_population])
                self.budget -= self.pop_size

                # Combine original and opposition populations
                combined_population = np.vstack((population, opposition_population))
                combined_fitness = np.concatenate((fitness, opposition_fitness))

                # Select the best individuals to form the new population
                best_indices = np.argsort(combined_fitness)[:self.pop_size]
                population = combined_population[best_indices]
                fitness = combined_fitness[best_indices]

                best_index = np.argmin(fitness)
                if fitness[best_index] < self.f_opt:
                    self.f_opt = fitness[best_index]
                    self.x_opt = population[best_index]
                    self.stagnation_counter = 0
                    self.previous_best_fitness = self.f_opt
                else:
                    self.stagnation_counter +=1

            if self.budget <= 0:
                break
                
        return self.f_opt, self.x_opt