import numpy as np

class AdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size_min=20, pop_size_max=100, F_base=0.5, CR_base=0.7, archive_size=50, F_range=0.3, CR_range=0.3, p_archive=0.1, lr=0.1, restart_patience=50):
        self.budget = budget
        self.dim = dim
        self.pop_size_min = pop_size_min
        self.pop_size_max = pop_size_max
        self.pop_size = pop_size_max  # Start with a larger population size
        self.F_base = F_base
        self.CR_base = CR_base
        self.archive_size = archive_size
        self.F_range = F_range
        self.CR_range = CR_range
        self.archive = []
        self.archive_fitness = []
        self.p_archive = p_archive
        self.lr = lr  # Learning rate for F and CR adaptation
        self.restart_patience = restart_patience  # Patience for stagnation detection
        self.stagnation_counter = 0

        self.successful_F = []
        self.successful_CR = []


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

            for i in range(self.pop_size):
                # Adaptive F and CR with learning rate
                if self.successful_F:
                    F = np.clip(np.mean(self.successful_F) + self.lr * np.random.randn(), 0.1, 1.0)
                else:
                    F = self.F_base + np.random.uniform(-self.F_range, self.F_range)
                    F = np.clip(F, 0.1, 1.0)
                
                if self.successful_CR:
                    CR = np.clip(np.mean(self.successful_CR) + self.lr * np.random.randn(), 0.1, 1.0)
                else:
                    CR = self.CR_base + np.random.uniform(-self.CR_range, self.CR_range)
                    CR = np.clip(CR, 0.1, 1.0)

                # Mutation Strategy: Adaptive selection
                rand = np.random.rand()
                if rand < 0.33:
                    # DE/rand/1
                    indices = np.random.choice(self.pop_size, 3, replace=False)
                    x1, x2, x3 = population[indices]
                    mutant = x1 + F * (x2 - x3)
                elif rand < 0.66:
                    # DE/current-to-best/1
                    indices = np.random.choice(self.pop_size, 2, replace=False)
                    x1, x2 = population[indices]
                    mutant = population[i] + F * (self.x_opt - population[i]) + F * (x1 - x2)
                else:
                    # DE/best/1
                    indices = np.random.choice(self.pop_size, 2, replace=False)
                    x1, x2 = population[indices]
                    mutant = self.x_opt + F * (x1 - x2)


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

                    # Update successful F and CR
                    self.successful_F.append(F)
                    self.successful_CR.append(CR)

                    # Add replaced vector to archive (probabilistic strategy)
                    if np.random.rand() < 0.5:
                        if len(self.archive) < self.archive_size:
                            self.archive.append(population[i])
                            self.archive_fitness.append(fitness[i])
                        else:
                            if self.archive_fitness: #Ensure archive not empty
                                worst_archive_index = np.argmax(self.archive_fitness)
                                self.archive[worst_archive_index] = population[i]
                                self.archive_fitness[worst_archive_index] = fitness[i]


                    # Update optimal solution
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

            population = new_population
            fitness = new_fitness

            # Clear successful F and CR every few generations
            if generation % 20 == 0:
                self.successful_F = []
                self.successful_CR = []

            # Check for stagnation and potentially restart or reduce population size
            if np.std(fitness) < 1e-6:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0

            if self.stagnation_counter > self.restart_patience:
                #Option 1: Restart
                population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
                fitness = np.array([func(x) for x in population])
                self.budget -= self.pop_size
                best_index = np.argmin(fitness)
                if fitness[best_index] < self.f_opt:
                    self.f_opt = fitness[best_index]
                    self.x_opt = population[best_index]
                self.stagnation_counter = 0

                #Option 2: Reduce population size (if possible)
                #self.pop_size = max(self.pop_size_min, int(self.pop_size * 0.8)) #Reduce by 20%
                #population = population[:self.pop_size] #Truncate
                #fitness = fitness[:self.pop_size]
                #print("Reduced population size to:", self.pop_size)

            if self.budget <= 0:
                break
                
        return self.f_opt, self.x_opt