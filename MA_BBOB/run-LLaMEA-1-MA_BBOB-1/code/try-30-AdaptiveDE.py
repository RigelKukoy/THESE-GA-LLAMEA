import numpy as np

class AdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, F_init=0.5, CR_init=0.7, archive_size=50, p_archive=0.1, ortho_group_size=5):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F_init
        self.CR = CR_init
        self.archive_size = archive_size
        self.archive = []
        self.p_archive = p_archive
        self.ortho_group_size = ortho_group_size
        self.F_ema_alpha = 0.1
        self.CR_ema_alpha = 0.1


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

        while self.budget > 0:
            new_population = np.copy(population)
            new_fitness = np.copy(fitness)

            for i in range(self.pop_size):
                # Mutation
                indices = np.random.choice(self.pop_size, 3, replace=False)
                x1, x2, x3 = population[indices]

                # Use archive with probability p_archive
                if np.random.rand() < self.p_archive and self.archive:
                    x4 = self.archive[np.random.randint(len(self.archive))]
                    mutant = x1 + self.F * (x2 - x3) + self.F * (x4 - population[i]) #Increased exploration by incorporating current individual to avoid premature convergence
                else:
                    mutant = x1 + self.F * (x2 - x3)
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                # Orthogonal Crossover
                trial = np.copy(population[i])
                group_indices = np.random.choice(self.dim, min(self.ortho_group_size, self.dim), replace=False)  # Select a subset of dimensions

                for j in group_indices:
                    if np.random.rand() < self.CR:
                        trial[j] = mutant[j]

                # Selection
                f_trial = func(trial)
                self.budget -= 1

                if f_trial < fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = f_trial

                    # Update F and CR using exponential moving average
                    self.F = (1 - self.F_ema_alpha) * self.F + self.F_ema_alpha * self.F #No new F since we dont save succesful ones.
                    self.CR = (1 - self.CR_ema_alpha) * self.CR + self.CR_ema_alpha * self.CR #Same for CR

                    # Simplified Archive Update: Add to archive if better than worst in archive
                    if len(self.archive) < self.archive_size:
                        self.archive.append(population[i])
                    else:
                        fitnesses = [func(x) for x in self.archive]
                        worst_index = np.argmax(fitnesses)
                        if fitness[i] < fitnesses[worst_index]:
                            self.archive[worst_index] = population[i]  # Replace worst with current

                    # Update optimal solution
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                else:
                    if len(self.archive) < self.archive_size:
                        self.archive.append(trial)
                    else:
                        fitnesses = [func(x) for x in self.archive]
                        worst_index = np.argmax(fitnesses)
                        if f_trial < fitnesses[worst_index]:
                            self.archive[worst_index] = trial  # Replace worst with current

            population = new_population
            fitness = new_fitness

            if self.budget <= 0:
                break
                
        return self.f_opt, self.x_opt