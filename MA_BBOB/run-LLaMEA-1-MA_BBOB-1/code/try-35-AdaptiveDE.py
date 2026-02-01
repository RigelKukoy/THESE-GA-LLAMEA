import numpy as np

class AdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, archive_size=100, F_init=0.5, CR_init=0.7):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.archive_size = archive_size
        self.F_init = F_init
        self.CR_init = CR_init
        self.F_memory = np.ones(self.pop_size) * self.F_init
        self.CR_memory = np.ones(self.pop_size) * self.CR_init
        self.archive = []
        self.archive_fitness = []
        self.success_F = []
        self.success_CR = []

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
        stagnation_counter = 0
        prev_best_fitness = self.f_opt

        while self.budget > 0:
            generation += 1
            new_population = np.copy(population)
            new_fitness = np.copy(fitness)
            successful_mutations = 0

            for i in range(self.pop_size):
                # Adaptive F and CR
                F = np.clip(np.random.normal(0.5, 0.3), 0.1, 1.0)
                CR = np.clip(np.random.normal(0.7, 0.1), 0.0, 1.0)

                # Mutation
                indices = np.random.choice(self.pop_size, 2, replace=False)
                x1, x2 = population[indices]

                # Utilize archive with dynamic probability
                archive_prob = min(0.4, generation / 500)  # Increase archive usage over time
                if len(self.archive) > 0 and np.random.rand() < archive_prob:
                     archive_index = np.random.randint(len(self.archive))
                     x3 = self.archive[archive_index]
                else:
                    indices = np.random.choice(self.pop_size, 1, replace=False)
                    x3 = population[indices[0]]

                # Weighted difference vector
                mutant = population[i] + F * (x1 - x2) + F * 0.5 * (x3 - population[i])  # Reduced pull to archive
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                # Orthogonal Crossover
                trial = np.copy(population[i])
                orthogonal_matrix = self.generate_orthogonal_array(self.dim)
                for j in range(self.dim):
                    if orthogonal_matrix[0, j] == 1:
                        trial[j] = mutant[j]
                
                # Selection
                f_trial = func(trial)
                self.budget -= 1
                if f_trial < fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = f_trial
                    successful_mutations += 1
                    self.success_F.append(F)
                    self.success_CR.append(CR)


                    # Dynamic archive management: replace worst in archive, only if improvement is significant
                    if len(self.archive) < self.archive_size or fitness[i] - f_trial > 1e-5:
                        if len(self.archive) < self.archive_size:
                            self.archive.append(population[i])
                            self.archive_fitness.append(fitness[i])
                        else:
                            worst_archive_index = np.argmax(self.archive_fitness)
                            if fitness[i] > f_trial:
                                self.archive[worst_archive_index] = population[i]
                                self.archive_fitness[worst_archive_index] = fitness[i]

                    # Update optimal solution
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                        stagnation_counter = 0 # Reset stagnation counter
                else:
                    # Dynamic archive management (trial vector)
                    if len(self.archive) < self.archive_size:
                        self.archive.append(trial)
                        self.archive_fitness.append(f_trial)
                    else:
                        worst_archive_index = np.argmax(self.archive_fitness)
                        if f_trial < self.archive_fitness[worst_archive_index]:
                            self.archive[worst_archive_index] = trial
                            self.archive_fitness[worst_archive_index] = f_trial

            # Update F and CR memory based on successful mutations (if any)
            if len(self.success_F) > 0:
                self.F_memory[i] = np.mean(self.success_F)
                self.CR_memory[i] = np.mean(self.success_CR)
                self.success_F = []
                self.success_CR = []


            population = new_population
            fitness = new_fitness

            # Restart population if stagnating
            if abs(self.f_opt - prev_best_fitness) < 1e-8: #More robust stagnation detection
                stagnation_counter += 1
            else:
                stagnation_counter = 0
            prev_best_fitness = self.f_opt

            if stagnation_counter > 100: #Increased stagnation threshold
                population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
                fitness = np.array([func(x) for x in population])
                self.budget -= self.pop_size
                best_index = np.argmin(fitness)
                if fitness[best_index] < self.f_opt:
                    self.f_opt = fitness[best_index]
                    self.x_opt = population[best_index]
                stagnation_counter = 0  # Reset after restart
                self.archive = [] #Clear archive after restart
                self.archive_fitness = []

            if self.budget <= 0:
                break
                
        return self.f_opt, self.x_opt

    def generate_orthogonal_array(self, n):
        # Use a simplified Hadamard matrix approach for demonstration.
        # For larger n, consider using dedicated orthogonal array libraries.
        if n <= 1:
            return np.ones((1, n))
        if n == 2:
            return np.array([[1, 1], [1, -1]])
        
        # For dimensions > 2, creating an orthogonal array becomes more complex
        # A simple strategy is to create a random binary array
        orthogonal_array = np.random.randint(0, 2, size=(1, n))
        return orthogonal_array