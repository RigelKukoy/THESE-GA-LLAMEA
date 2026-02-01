import numpy as np
from scipy.stats import norm

class AdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, F_base=0.5, CR_base=0.7, archive_size=100, F_range=0.3, CR_range=0.3, orthogonal_design_size=5):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.min_pop_size = 10  # Minimum population size
        self.max_pop_size = 100 # Maximum population size
        self.F_base = F_base
        self.CR_base = CR_base
        self.archive_size = archive_size
        self.F_range = F_range
        self.CR_range = CR_range
        self.archive = []
        self.archive_fitness = []
        self.orthogonal_design_size = orthogonal_design_size
        self.success_memory_F = []  # Store successful F values
        self.success_memory_CR = []  # Store successful CR values
        self.levy_exponent = 1.5  # Exponent for Levy flight

    def levy_flight(self, size):
        """Generate Levy flight steps."""
        u = np.random.randn(size)
        v = np.random.randn(size)
        step = (u * np.sqrt(np.pi / 2) * np.gamma(1 + self.levy_exponent) * np.sin(np.pi * self.levy_exponent / 2)) / \
               (np.abs(v) ** (1 / self.levy_exponent) * np.gamma((1 + self.levy_exponent) / 2) * self.levy_exponent * 2 ** ((self.levy_exponent - 1) / 2))
        return step

    def orthogonal_design(self):
        """Generate orthogonal design for F and CR parameters."""
        design = np.zeros((self.orthogonal_design_size, 2))
        for i in range(self.orthogonal_design_size):
            design[i, 0] = np.random.uniform(self.F_base - self.F_range, self.F_base + self.F_range)
            design[i, 1] = np.random.uniform(self.CR_base - self.CR_range, self.CR_base + self.CR_range)
            design[i, 0] = np.clip(design[i, 0], 0.1, 1.0)
            design[i, 1] = np.clip(design[i, 1], 0.1, 1.0)
        return design

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

        generation = 0
        while self.budget > 0:
            generation += 1
            new_population = np.copy(population)
            new_fitness = np.copy(fitness)

            orthogonal_params = self.orthogonal_design()

            for i in range(self.pop_size):
                # Orthogonal design parameter selection
                design_index = i % self.orthogonal_design_size
                F = orthogonal_params[design_index, 0]
                CR = orthogonal_params[design_index, 1]

                # Mutation
                indices = np.random.choice(self.pop_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant = x1 + F * (x2 - x3)

                # Levy flight to enhance exploration
                levy_steps = self.levy_flight(self.dim) * 0.01  # Scale Levy steps
                mutant += levy_steps
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                # Crossover
                trial = np.copy(population[i])
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                
                # Selection
                f_trial = func(trial)
                self.budget -= 1
                if f_trial < fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = f_trial

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

                    # Store successful F and CR values
                    self.success_memory_F.append(F)
                    self.success_memory_CR.append(CR)

                else:
                     # Add trial vector to archive (combined strategy)
                    if len(self.archive) < self.archive_size:
                        self.archive.append(trial)
                        self.archive_fitness.append(f_trial)
                    else:
                        # Replace worst in archive
                        worst_archive_index = np.argmax(self.archive_fitness)
                        self.archive[worst_archive_index] = trial
                        self.archive_fitness[worst_archive_index] = f_trial
                        

            population = new_population
            fitness = new_fitness

            # Dynamic population size adjustment
            if generation % 20 == 0:
                if len(self.success_memory_F) > 10:
                    mean_success_F = np.mean(self.success_memory_F)
                    mean_success_CR = np.mean(self.success_memory_CR)
                    
                    # Adjust population size based on success
                    if mean_success_F > self.F_base and mean_success_CR > self.CR_base:
                        self.pop_size = min(self.pop_size + 5, self.max_pop_size)
                    elif mean_success_F < self.F_base and mean_success_CR < self.CR_base:
                        self.pop_size = max(self.pop_size - 5, self.min_pop_size)
                    
                    # Reinitialize population if size changed
                    if population.shape[0] != self.pop_size:
                         population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
                         fitness = np.array([func(x) for x in population])
                         self.budget -= self.pop_size

                    self.success_memory_F = []  # Clear memory
                    self.success_memory_CR = []

            # Restart population if stagnating
            if generation % 50 == 0:
                if np.std(fitness) < 1e-6:  # Stagnation criterion
                    population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
                    fitness = np.array([func(x) for x in population])
                    self.budget -= self.pop_size
                    best_index = np.argmin(fitness)
                    if fitness[best_index] < self.f_opt:
                        self.f_opt = fitness[best_index]
                        self.x_opt = population[best_index]

            if self.budget <= 0:
                break
                
        return self.f_opt, self.x_opt