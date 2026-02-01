import numpy as np

class AdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, archive_size=100, F_init=0.5, CR_init=0.7, ring_neighbors=3):
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
        self.ring_neighbors = ring_neighbors

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

            for i in range(self.pop_size):
                # Adaptive F and CR
                self.F_memory[i] = np.clip(np.random.normal(0.5, 0.3), 0.1, 1.0)
                self.CR_memory[i] = np.clip(np.random.normal(0.7, 0.1), 0.0, 1.0)

                # Mutation - Ring Topology
                neighbors = [(i + j) % self.pop_size for j in range(1, self.ring_neighbors + 1)]
                indices = np.random.choice(neighbors, 2, replace=False)
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
                mutant = population[i] + self.F_memory[i] * (x1 - x2) + self.F_memory[i] * 0.5 * (x3 - population[i])  # Reduced pull to archive
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                # Orthogonal Learning
                if np.random.rand() < 0.1:  # Probability of applying orthogonal learning
                    H = self._generate_orthogonal_array(self.dim)
                    orthogonal_trials = []
                    for h in H:
                        trial = population[i] + 0.1 * h * (func.bounds.ub - func.bounds.lb) #Scale orthogonal adjustments
                        trial = np.clip(trial, func.bounds.lb, func.bounds.ub)
                        orthogonal_trials.append(trial)

                    orthogonal_fitness = [func(trial) for trial in orthogonal_trials]
                    self.budget -= len(orthogonal_trials)
                    best_orthogonal_index = np.argmin(orthogonal_fitness)
                    if orthogonal_fitness[best_orthogonal_index] < fitness[i]:
                        mutant = orthogonal_trials[best_orthogonal_index]
                        

                # Crossover
                trial = np.copy(population[i])
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.CR_memory[i] or j == j_rand:
                        trial[j] = mutant[j]
                
                # Selection
                f_trial = func(trial)
                self.budget -= 1
                if f_trial < fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = f_trial

                    # Dynamic archive management: replace worst in archive
                    combined_population = np.concatenate((population, np.array(self.archive)))
                    combined_fitness = np.concatenate((fitness, np.array(self.archive_fitness)))

                    if len(self.archive) < self.archive_size:
                        self.archive.append(population[i])
                        self.archive_fitness.append(fitness[i])
                    else:
                        worst_archive_index = np.argmax(self.archive_fitness)
                        if fitness[i] < self.archive_fitness[worst_archive_index]:
                            self.archive[worst_archive_index] = population[i]
                            self.archive_fitness[worst_archive_index] = fitness[i]

                    # Update optimal solution
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                        stagnation_counter = 0 # Reset stagnation counter
                else:
                    # Dynamic archive management: replace worst in archive (trial vector)
                    combined_population = np.concatenate((population, np.array(self.archive)))
                    combined_fitness = np.concatenate((fitness, np.array(self.archive_fitness)))
                    if len(self.archive) < self.archive_size:
                        self.archive.append(trial)
                        self.archive_fitness.append(f_trial)
                    else:
                        worst_archive_index = np.argmax(self.archive_fitness)
                        if f_trial < self.archive_fitness[worst_archive_index]:
                            self.archive[worst_archive_index] = trial
                            self.archive_fitness[worst_archive_index] = f_trial
                        

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

    def _generate_orthogonal_array(self, dim):
        # Simple 2-level orthogonal array for demonstration (L8)
        if dim <= 1:
            return [[-1], [1]]
        if dim <= 3:
            return [
                [-1, -1, -1],
                [ 1, -1, -1],
                [-1,  1, -1],
                [ 1,  1, -1],
                [-1, -1,  1],
                [ 1, -1,  1],
                [-1,  1,  1],
                [ 1,  1,  1]
            ]
        
        #For larger dimensions we will simply generate random binary combinations 
        num_points = 2 * dim  #Generate 2x the number of dims
        orthogonal_array = np.random.choice([-1, 1], size=(num_points, dim))
        return orthogonal_array.tolist()