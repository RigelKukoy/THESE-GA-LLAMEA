import numpy as np

class AdaptiveArchiveDE:
    def __init__(self, budget=10000, dim=10, initial_popsize=50, archive_size=10, F=0.5, CR=0.7):
        self.budget = budget
        self.dim = dim
        self.popsize = initial_popsize
        self.archive_size = archive_size
        self.F = F
        self.CR = CR
        self.population = None
        self.fitness = None
        self.archive = None
        self.archive_fitness = None
        self.eval_count = 0
        self.f_opt = np.inf
        self.x_opt = None

    def initialize(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.popsize, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.eval_count += self.popsize
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]
        self.archive = np.zeros((self.archive_size, self.dim))
        self.archive_fitness = np.full(self.archive_size, np.inf)

    def ensure_bounds(self, vec, lb, ub):
        vec_clipped = np.clip(vec, lb, ub)
        return vec_clipped
        
    def mutate(self, target_index):
        indices = [i for i in range(self.popsize) if i != target_index]
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = self.population[a] + self.F * (self.population[b] - self.population[c])
        return mutant

    def crossover(self, mutant, target, CR):
        cross_points = np.random.rand(self.dim) < CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def update_archive(self, individual, fitness_value):
        if fitness_value < np.max(self.archive_fitness):
            worst_index = np.argmax(self.archive_fitness)
            self.archive[worst_index] = individual
            self.archive_fitness[worst_index] = fitness_value
            
    def __call__(self, func):
        self.initialize(func)

        while self.eval_count < self.budget:
            for i in range(self.popsize):
                
                mutant = self.mutate(i)
                trial = self.crossover(mutant, self.population[i], self.CR)
                trial = self.ensure_bounds(trial, func.bounds.lb, func.bounds.ub)

                f = func(trial)
                self.eval_count += 1

                if f < self.fitness[i]:
                    self.update_archive(self.population[i], self.fitness[i])
                    self.population[i] = trial
                    self.fitness[i] = f

                    if f < self.f_opt:
                        self.f_opt = f
                        self.x_opt = trial
                else:
                    # Archive exploitation: replace a random individual in the population 
                    # with a random member from the archive with a small probability
                    if np.random.rand() < 0.05 and self.eval_count < 0.9 * self.budget: 
                       if np.any(self.archive_fitness < np.inf):
                           valid_indices = np.where(self.archive_fitness < np.inf)[0]
                           arch_idx = np.random.choice(valid_indices)
                           self.population[np.random.randint(self.popsize)] = self.archive[arch_idx]
                           self.fitness = np.array([func(x) for x in self.population])
                           self.eval_count += self.popsize -1 # Correction: minus one as only popsize-1 new calls where made. 
                           
                           if np.min(self.fitness) < self.f_opt: # Find if any new bests. 
                               self.f_opt = np.min(self.fitness)
                               self.x_opt = self.population[np.argmin(self.fitness)]   
                
                # Adjust population size based on progress
                if self.eval_count % (self.dim * 5) == 0:
                    if np.std(self.fitness) < 1e-6 and self.popsize < 100:
                        self.popsize = min(self.popsize + 5, 100)  # Increase popsize if converged
                        new_individuals = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(5, self.dim))
                        new_fitness = np.array([func(x) for x in new_individuals])
                        self.eval_count += 5
                        
                        self.population = np.vstack((self.population, new_individuals))
                        self.fitness = np.concatenate((self.fitness, new_fitness))

                    elif self.popsize > 20 and self.eval_count < 0.75 * self.budget:
                        self.popsize = max(20, self.popsize - 2)  # Decrease popsize if not making progress
                        self.population = self.population[:self.popsize]
                        self.fitness = self.fitness[:self.popsize]
                    

                if self.eval_count >= self.budget:
                    break

        return self.f_opt, self.x_opt