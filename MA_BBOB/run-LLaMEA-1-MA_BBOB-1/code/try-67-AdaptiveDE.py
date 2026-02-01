import numpy as np
from sklearn.cluster import KMeans

class AdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size_init=50, archive_size_init=50, F_init=0.5, CR_init=0.7, num_clusters=5, diversity_threshold=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size_init = pop_size_init
        self.pop_size = pop_size_init  # Dynamic population size
        self.archive_size_init = archive_size_init
        self.archive_size = archive_size_init
        self.F_init = F_init
        self.CR_init = CR_init
        self.F_memory = np.ones(self.pop_size_init) * self.F_init
        self.CR_memory = np.ones(self.pop_size_init) * self.CR_init
        self.archive = []
        self.archive_fitness = []
        self.archive_age = []
        self.success_F = []
        self.success_CR = []
        self.success_count = 0
        self.failure_count = 0
        self.num_clusters = num_clusters  # Number of clusters for archive management
        self.diversity_threshold = diversity_threshold
        self.pop_size_min = 10
        self.pop_size_max = 100


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
        archive_clear_interval = 200

        while self.budget > 0:
            generation += 1
            new_population = np.copy(population)
            new_fitness = np.copy(fitness)

            # Adjust archive size dynamically
            success_rate = self.success_count / (self.success_count + self.failure_count + 1e-9)
            self.archive_size = int(self.archive_size_init * (0.5 + success_rate))

            # Adjust population size dynamically
            if success_rate > 0.6:
                self.pop_size = min(self.pop_size + 1, self.pop_size_max)
            elif success_rate < 0.2:
                self.pop_size = max(self.pop_size - 1, self.pop_size_min)

            if self.pop_size != population.shape[0]:
                population = population[:min(self.pop_size,population.shape[0])]
                fitness = fitness[:min(self.pop_size,fitness.shape[0])]
                new_population = population
                new_fitness = fitness

            for i in range(self.pop_size):
                # Adaptive F and CR
                if self.success_F:
                    F_mean = np.mean(self.success_F)
                    self.F_memory[i] = np.clip(np.random.normal(F_mean, 0.3), 0.1, 1.0)
                else:
                    self.F_memory[i] = np.clip(np.random.normal(0.5, 0.3), 0.1, 1.0)

                if self.success_CR:
                    CR_mean = np.mean(self.success_CR)
                    self.CR_memory[i] = np.clip(np.random.normal(CR_mean, 0.1), 0.0, 1.0)
                else:
                    self.CR_memory[i] = np.clip(np.random.normal(0.7, 0.1), 0.0, 1.0)

                # Mutation Strategies with Probabilistic Selection
                mutation_strategy = np.random.choice([1, 2, 3], p=[0.4, 0.3, 0.3])  # Adjusted probabilities

                if mutation_strategy == 1:
                    # DE/rand/1
                    indices = np.random.choice(self.pop_size, 3, replace=False)
                    x1, x2, x3 = population[indices]
                    mutant = x1 + self.F_memory[i] * (x2 - x3)
                elif mutation_strategy == 2:
                    # DE/current-to-best/1
                    indices = np.random.choice(self.pop_size, 2, replace=False)
                    x1, x2 = population[indices]
                    mutant = population[i] + self.F_memory[i] * (self.x_opt - population[i]) + self.F_memory[i] * (x1 - x2)
                else:
                    # DE/rand/2 with archive
                    indices = np.random.choice(self.pop_size, 2, replace=False)
                    x1, x2 = population[indices]
                    if len(self.archive) > 0:
                        archive_index = np.random.randint(len(self.archive))
                        x3 = self.archive[archive_index]
                    else:
                        x3 = population[np.random.choice(self.pop_size)]
                    x4 = population[np.random.choice(self.pop_size)]
                    mutant = x1 + self.F_memory[i] * (x2 - x3) + self.F_memory[i] * (x4 - population[i])

                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                # Orthogonal Crossover
                trial = np.copy(population[i])
                num_changed_vars = 0
                for j in range(self.dim):
                    if np.random.rand() < self.CR_memory[i]:
                        trial[j] = mutant[j]
                        num_changed_vars += 1
                if num_changed_vars == 0:
                    j_rand = np.random.randint(self.dim)
                    trial[j_rand] = mutant[j_rand]

                # Selection
                f_trial = func(trial)
                self.budget -= 1
                if f_trial < fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = f_trial

                    self.success_F.append(self.F_memory[i])
                    self.success_CR.append(self.CR_memory[i])
                    if len(self.success_F) > 10:
                        self.success_F.pop(0)
                        self.success_CR.pop(0)
                    self.success_count += 1

                    # Dynamic archive management with crowding distance
                    if len(self.archive) < self.archive_size:
                        self.archive.append(population[i])
                        self.archive_fitness.append(fitness[i])
                        self.archive_age.append(0)
                    else:
                        if len(self.archive) > 0:
                            # Crowding distance calculation (simplified)
                            distances = [np.linalg.norm(population[i] - archive_member) for archive_member in self.archive]
                            min_distance_index = np.argmin(distances)

                            if fitness[i] < self.archive_fitness[min_distance_index]:
                                self.archive[min_distance_index] = population[i]
                                self.archive_fitness[min_distance_index] = fitness[i]
                                self.archive_age[min_distance_index] = 0

                    # Update optimal solution
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                        stagnation_counter = 0
                else:
                    self.failure_count += 1
                    # Dynamic archive management with crowding distance (trial vector)
                    if len(self.archive) < self.archive_size:
                        self.archive.append(trial)
                        self.archive_fitness.append(f_trial)
                        self.archive_age.append(0)
                    else:
                        if len(self.archive) > 0:
                            # Crowding distance calculation (simplified)
                            distances = [np.linalg.norm(trial - archive_member) for archive_member in self.archive]
                            min_distance_index = np.argmin(distances)

                            if f_trial < self.archive_fitness[min_distance_index]:
                                self.archive[min_distance_index] = trial
                                self.archive_fitness[min_distance_index] = f_trial
                                self.archive_age[min_distance_index] = 0

            # Update age of archive members
            self.archive_age = [age + 1 for age in self.archive_age]

            population = new_population
            fitness = new_fitness

            # Restart population if stagnating (simplified diversity check)
            diversity = np.std(population)
            if diversity < self.diversity_threshold:
                population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
                fitness = np.array([func(x) for x in population])
                self.budget -= self.pop_size
                best_index = np.argmin(fitness)
                if fitness[best_index] < self.f_opt:
                    self.f_opt = fitness[best_index]
                    self.x_opt = population[best_index]
                stagnation_counter = 0
                self.archive = []
                self.archive_fitness = []
                self.archive_age = []
                self.success_F = []
                self.success_CR = []
                self.success_count = 0
                self.failure_count = 0
                self.F_memory = np.ones(self.pop_size) * self.F_init
                self.CR_memory = np.ones(self.pop_size) * self.CR_init

            # Periodic archive clearing and clustering
            if generation % archive_clear_interval == 0 and len(self.archive) > 0:
                # Clustering to maintain diversity
                try:
                    kmeans = KMeans(n_clusters=min(self.num_clusters, len(self.archive)), random_state=0, n_init=10)  # Ensure n_clusters <= n_samples
                    kmeans.fit(self.archive)
                    cluster_labels = kmeans.labels_

                    # Keep only the best individual from each cluster
                    new_archive = []
                    new_archive_fitness = []
                    new_archive_age = []
                    for cluster_id in range(kmeans.n_clusters):
                        cluster_indices = np.where(cluster_labels == cluster_id)[0]
                        best_index_in_cluster = cluster_indices[np.argmin([self.archive_fitness[i] for i in cluster_indices])]
                        new_archive.append(self.archive[best_index_in_cluster])
                        new_archive_fitness.append(self.archive_fitness[best_index_in_cluster])
                        new_archive_age.append(0)  # Reset age

                    self.archive = new_archive
                    self.archive_fitness = new_archive_fitness
                    self.archive_age = new_archive_age
                except ValueError as e:
                    print(f"KMeans clustering failed: {e}")
                    self.archive = []
                    self.archive_fitness = []
                    self.archive_age = []

            if self.budget <= 0:
                break

        return self.f_opt, self.x_opt