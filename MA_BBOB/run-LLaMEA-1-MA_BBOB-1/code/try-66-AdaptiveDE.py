import numpy as np
from scipy.optimize import minimize

class AdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size_min=20, pop_size_max=100, F_base=0.5, CR_base=0.7, archive_size=100, F_range=0.3, CR_range=0.3, orthogonal_learning_rate=0.1, restart_patience=50, archive_prob=0.1, current_to_best_prob=0.2, gamma_F = 0.1, gamma_CR = 0.1, local_search_frequency=25):
        self.budget = budget
        self.dim = dim
        self.pop_size_min = pop_size_min
        self.pop_size_max = pop_size_max
        self.pop_size = pop_size_max
        self.F_base = F_base
        self.CR_base = CR_base
        self.archive_size = archive_size
        self.F_range = F_range
        self.CR_range = CR_range
        self.archive = []
        self.archive_fitness = []
        self.orthogonal_learning_rate = orthogonal_learning_rate
        self.restart_patience = restart_patience
        self.stagnation_counter = 0
        self.previous_best_fitness = np.Inf
        self.F_success_history = []
        self.CR_success_history = []
        self.F_failure_history = []
        self.CR_failure_history = []
        self.archive_prob = archive_prob
        self.current_to_best_prob = current_to_best_prob
        self.gamma_F = gamma_F
        self.gamma_CR = gamma_CR
        self.local_search_frequency = local_search_frequency
        self.generation = 0

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

        while self.budget > 0:
            self.generation += 1
            new_population = np.copy(population)
            new_fitness = np.copy(fitness)
            
            # Self-adaptive F and CR
            if self.F_success_history:
                self.F_base = (1-self.gamma_F) * self.F_base + self.gamma_F * np.mean(self.F_success_history)
                self.F_base = np.clip(self.F_base, 0.1, 1.0)
            if self.CR_success_history:
                self.CR_base = (1-self.gamma_CR) * self.CR_base + self.gamma_CR * np.mean(self.CR_success_history)
                self.CR_base = np.clip(self.CR_base, 0.1, 1.0)

            for i in range(self.pop_size):
                # Adaptive F and CR
                F = self.F_base + np.random.uniform(-self.F_range, self.F_range)
                CR = self.CR_base + np.random.uniform(-self.CR_range, self.CR_range)
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
                if len(self.archive) > 0 and np.random.rand() < self.archive_prob:  # archive_prob chance to use archive
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
                    
                    # Record successful F and CR values
                    self.F_success_history.append(F)
                    self.CR_success_history.append(CR)
                    if len(self.F_success_history) > 50:
                        self.F_success_history.pop(0)
                        self.CR_success_history.pop(0)
                    if self.F_failure_history:
                        self.F_failure_history.pop(0)
                    if self.CR_failure_history:
                        self.CR_failure_history.pop(0)


                    # Add replaced vector to archive (combined strategy)
                    if len(self.archive) < self.archive_size:
                        self.archive.append(population[i])
                        self.archive_fitness.append(fitness[i])
                    else:
                         # Replace worst in archive
                        worst_archive_index = np.argmax(self.archive_fitness)
                        self.archive[worst_archive_index] = population[i]
                        self.archive_fitness[worst_archive_index] = fitness[i]

                    # Update optimal solution
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                        self.stagnation_counter = 0
                        self.previous_best_fitness = self.f_opt
                else:
                    self.F_failure_history.append(F)
                    self.CR_failure_history.append(CR)
                    if len(self.F_failure_history) > 50:
                         self.F_failure_history.pop(0)
                    if len(self.CR_failure_history) > 50:
                         self.CR_failure_history.pop(0)
                    if self.F_success_history:
                        self.F_success_history.pop(0)
                    if self.CR_success_history:
                        self.CR_success_history.pop(0)

                     # Add trial vector to archive (combined strategy)
                    if len(self.archive) < self.archive_size:
                        self.archive.append(trial)
                        self.archive_fitness.append(f_trial)
                    else:
                        # Replace worst in archive
                        worst_archive_index = np.argmax(self.archive_fitness)
                        if f_trial < self.archive_fitness[worst_archive_index]:
                            self.archive[worst_archive_index] = trial
                            self.archive_fitness[worst_archive_index] = f_trial
                        

            population = new_population
            fitness = new_fitness
            
            # Adjust population size
            if self.generation % 10 == 0:
                if len(self.F_success_history) > len(self.F_failure_history):
                    self.pop_size = min(self.pop_size + 5, self.pop_size_max)
                else:
                    self.pop_size = max(self.pop_size - 5, self.pop_size_min)
                
                # Regenerate population with new size
                population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
                fitness = np.array([func(x) for x in population])
                self.budget -= self.pop_size
                best_index = np.argmin(fitness)
                if fitness[best_index] < self.f_opt:
                    self.f_opt = fitness[best_index]
                    self.x_opt = population[best_index]

            # Local search intensification
            if self.generation % self.local_search_frequency == 0:
                bounds = [(func.bounds.lb, func.bounds.ub)] * self.dim
                result = minimize(func, self.x_opt, method='Nelder-Mead', bounds=bounds, options={'maxfev': min(500, self.budget)})
                if result.fun < self.f_opt:
                    self.f_opt = result.fun
                    self.x_opt = result.x
                self.budget -= result.nfev


            # Restart population if stagnating
            if self.stagnation_counter > self.restart_patience:
                population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
                fitness = np.array([func(x) for x in population])
                self.budget -= self.pop_size
                best_index = np.argmin(fitness)
                if fitness[best_index] < self.f_opt:
                    self.f_opt = fitness[best_index]
                    self.x_opt = population[best_index]
                    self.stagnation_counter = 0
                    self.previous_best_fitness = self.f_opt
                else:
                    self.stagnation_counter +=1
                    
                # Reset F and CR history
                self.F_success_history = []
                self.CR_success_history = []
                self.F_failure_history = []
                self.CR_failure_history = []
                

            if self.budget <= 0:
                break
                
        return self.f_opt, self.x_opt