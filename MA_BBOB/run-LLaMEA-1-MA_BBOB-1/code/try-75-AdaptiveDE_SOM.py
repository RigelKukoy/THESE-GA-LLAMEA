import numpy as np

class AdaptiveDE_SOM:
    def __init__(self, budget=10000, dim=10, pop_size=50, F_base=0.5, CR_base=0.7, archive_size=100, F_range=0.3, CR_range=0.3, orthogonal_learning_rate=0.1, som_grid_size=10, som_learning_rate=0.1, som_sigma=1.0, aging_rate=0.05, niche_radius=0.5):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F_base = F_base
        self.CR_base = CR_base
        self.archive_size = archive_size
        self.F_range = F_range
        self.CR_range = CR_range
        self.archive = []
        self.archive_fitness = []
        self.orthogonal_learning_rate = orthogonal_learning_rate
        self.aging_rate = aging_rate
        self.F_history = []
        self.CR_history = []

        # SOM parameters
        self.som_grid_size = som_grid_size
        self.som_learning_rate = som_learning_rate
        self.som_sigma = som_sigma
        self.som_weights = np.random.rand(som_grid_size, som_grid_size, dim)  # SOM grid
        self.som_stagnation_threshold = 0.9  # Threshold for stagnation detection
        self.som_stagnation_patience = 3 #Number of generations to wait before restart

        self.stagnation_counter = 0
        self.previous_best_fitness = np.Inf
        self.restart_flag = False

        #Niche Parameters
        self.niche_radius = niche_radius


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
            self.previous_best_fitness = self.f_opt

        generation = 0
        while self.budget > 0:
            generation += 1
            new_population = np.copy(population)
            new_fitness = np.copy(fitness)

            # Adaptive F and CR using weighted historical success
            if self.F_history and self.CR_history:
                weights_F = np.linspace(0.1, 1.0, len(self.F_history))
                weights_CR = np.linspace(0.1, 1.0, len(self.CR_history))

                F_weighted_avg = np.average(self.F_history, weights=weights_F)
                CR_weighted_avg = np.average(self.CR_history, weights=weights_CR)
            else:
                F_weighted_avg = self.F_base
                CR_weighted_avg = self.CR_base

            for i in range(self.pop_size):
                # Adaptive F and CR
                F = F_weighted_avg + np.random.uniform(-self.F_range, self.F_range)
                CR = CR_weighted_avg + np.random.uniform(-self.CR_range, self.CR_range)
                F = np.clip(F, 0.1, 1.0)
                CR = np.clip(CR, 0.1, 1.0)

                # Mutation
                indices = np.random.choice(self.pop_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant = x1 + F * (x2 - x3)

                # Incorporate archive
                if len(self.archive) > 0 and np.random.rand() < 0.1:  # 10% chance to use archive
                    archive_index = np.random.randint(len(self.archive))
                    mutant = x1 + F * (x2 - self.archive[archive_index])

                # Repair mechanism
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

                # Niche Mechanism - encourage exploration in under-explored SOM regions
                bmu_index = self.find_bmu(population[i])
                nearby_individuals = [p for p in population if np.linalg.norm(p - population[i]) < self.niche_radius]
                if len(nearby_individuals) < self.pop_size / self.som_grid_size:  # Encourage exploration
                    som_node = self.som_weights[bmu_index[0], bmu_index[1]]
                    trial = 0.7 * trial + 0.3 * som_node + np.random.normal(0, 0.05, self.dim)
                    trial = np.clip(trial, func.bounds.lb, func.bounds.ub)


                # Selection
                f_trial = func(trial)
                self.budget -= 1
                if f_trial < fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = f_trial
                    
                    # Record successful F and CR values
                    self.F_history.append(F)
                    self.CR_history.append(CR)
                    if len(self.F_history) > 50:
                        self.F_history.pop(0)
                        self.CR_history.pop(0)

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
                        self.previous_best_fitness = self.f_opt
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

            # SOM training (diversity maintenance)
            self.train_som(population)
            diversity_metric = self.calculate_diversity_metric(population)
            
            if diversity_metric > self.som_stagnation_threshold:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0

            # Dynamic restart based on SOM-detected stagnation
            if self.stagnation_counter > self.som_stagnation_patience:
                population = self.initialize_population_from_som(func.bounds.lb, func.bounds.ub)
                fitness = np.array([func(x) for x in population])
                self.budget -= self.pop_size
                
                best_index = np.argmin(fitness)
                if fitness[best_index] < self.f_opt:
                    self.f_opt = fitness[best_index]
                    self.x_opt = population[best_index]
                    self.previous_best_fitness = self.f_opt
                
                self.F_history = []
                self.CR_history = []
                self.stagnation_counter = 0


            # Archive aging
            for k in range(len(self.archive_fitness)):
                self.archive_fitness[k] *= (1 - self.aging_rate)
            
            if self.budget <= 0:
                break
        return self.f_opt, self.x_opt

    def train_som(self, population):
        """Trains the Self-Organizing Map (SOM) with the current population."""
        for x in population:
            # Find the best matching unit (BMU)
            bmu_index = self.find_bmu(x)

            # Update the SOM weights around the BMU
            for i in range(self.som_grid_size):
                for j in range(self.som_grid_size):
                    distance = np.sqrt((i - bmu_index[0])**2 + (j - bmu_index[1])**2)
                    influence = np.exp(-distance**2 / (2 * self.som_sigma**2))
                    self.som_weights[i, j] += self.som_learning_rate * influence * (x - self.som_weights[i, j])

    def find_bmu(self, individual):
        """Finds the best matching unit (BMU) in the SOM for a given individual."""
        min_distance = np.inf
        bmu_index = None
        for i in range(self.som_grid_size):
            for j in range(self.som_grid_size):
                distance = np.linalg.norm(individual - self.som_weights[i, j])
                if distance < min_distance:
                    min_distance = distance
                    bmu_index = (i, j)
        return bmu_index

    def calculate_diversity_metric(self, population):
        """Calculates a diversity metric based on the SOM."""
        bmu_indices = [self.find_bmu(x) for x in population]
        unique_bmu_count = len(set(bmu_indices))
        return unique_bmu_count / self.pop_size

    def initialize_population_from_som(self, lower_bound, upper_bound):
        """Initializes a new population by sampling from the SOM."""
        new_population = np.zeros((self.pop_size, self.dim))
        for i in range(self.pop_size):
            # Randomly select a node from the SOM
            row = np.random.randint(0, self.som_grid_size)
            col = np.random.randint(0, self.som_grid_size)
            
            # Perturb the SOM node's weights to create a new individual
            new_individual = self.som_weights[row, col] + np.random.normal(0, 0.1, self.dim)
            new_individual = np.clip(new_individual, lower_bound, upper_bound)
            new_population[i] = new_individual
        return new_population