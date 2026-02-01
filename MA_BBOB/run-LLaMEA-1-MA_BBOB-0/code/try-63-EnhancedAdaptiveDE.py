import numpy as np

class EnhancedAdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size_min=20, pop_size_max=100, archive_size=15, F_initial=0.5, CR_initial=0.5, F_learning_rate=0.1, CR_learning_rate=0.1, stagnation_threshold=1000, ortho_group_size=5, exploration_prob=0.1, landscape_awareness=True):
        self.budget = budget
        self.dim = dim
        self.pop_size_min = pop_size_min
        self.pop_size_max = pop_size_max
        self.pop_size = pop_size_max
        self.archive_size = archive_size
        self.F = F_initial
        self.CR = CR_initial
        self.population = None
        self.fitness = None
        self.archive = []
        self.F_learning_rate = F_learning_rate
        self.CR_learning_rate = CR_learning_rate
        self.successful_F = []
        self.successful_CR = []
        self.F_momentum = 0.0
        self.CR_momentum = 0.0
        self.momentum_coeff = 0.9
        self.stagnation_threshold = stagnation_threshold
        self.stagnation_counter = 0
        self.best_fitness_history = []
        self.ortho_group_size = ortho_group_size
        self.pop_size_adaptation_rate = 0.1
        self.exploration_prob = exploration_prob
        self.landscape_awareness = landscape_awareness # Enable/disable landscape awareness
        self.neighborhood_size = 5 # Neighborhood size for landscape analysis


    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        self.archive = []

    def mutate(self):
        mutated_population = np.zeros_like(self.population)
        for i in range(self.pop_size):
            if self.landscape_awareness:
                # Analyze local landscape
                neighborhood_indices = np.random.choice(self.pop_size, self.neighborhood_size, replace=False)
                neighborhood_fitness = self.fitness[neighborhood_indices]
                std_fitness = np.std(neighborhood_fitness)

                # Adaptive F based on landscape variance
                adaptive_F = self.F * (1 + std_fitness)
                adaptive_F = np.clip(adaptive_F, 0.1, 1.0)  # Ensure F stays within reasonable bounds

                if np.random.rand() < 0.5: # Current-to-best mutation
                    best_idx = np.argmin(self.fitness)
                    idxs = np.random.choice(self.pop_size, 2, replace=False)
                    x1, x2 = self.population[idxs]
                    mutated_population[i] = self.population[i] + adaptive_F * (self.population[best_idx] - self.population[i]) + adaptive_F * (x1 - x2)
                elif np.random.rand() < self.exploration_prob and len(self.archive) > 0:  # Archive-based mutation
                    archive_idx = np.random.randint(len(self.archive))
                    idxs = np.random.choice(self.pop_size, 2, replace=False)
                    x1, x2 = self.population[idxs]
                    mutated_population[i] = self.archive[archive_idx] + adaptive_F * (x1 - x2)
                else: # Random mutation
                    idxs = np.random.choice(self.pop_size, 3, replace=False)
                    x1, x2, x3 = self.population[idxs]
                    mutated_population[i] = x1 + adaptive_F * (x2 - x3)
            else:
                if np.random.rand() < 0.5: # Current-to-best mutation
                    best_idx = np.argmin(self.fitness)
                    idxs = np.random.choice(self.pop_size, 2, replace=False)
                    x1, x2 = self.population[idxs]
                    mutated_population[i] = self.population[i] + self.F * (self.population[best_idx] - self.population[i]) + self.F * (x1 - x2)
                elif np.random.rand() < self.exploration_prob and len(self.archive) > 0:  # Archive-based mutation
                    archive_idx = np.random.randint(len(self.archive))
                    idxs = np.random.choice(self.pop_size, 2, replace=False)
                    x1, x2 = self.population[idxs]
                    mutated_population[i] = self.archive[archive_idx] + self.F * (x1 - x2)
                else: # Random mutation
                    idxs = np.random.choice(self.pop_size, 3, replace=False)
                    x1, x2, x3 = self.population[idxs]
                    mutated_population[i] = x1 + self.F * (x2 - x3)
        return mutated_population

    def crossover(self, mutated_population):
        crossed_population = np.zeros_like(self.population)
        for i in range(self.pop_size):
            group_idx = i // self.ortho_group_size
            start_idx = group_idx * self.ortho_group_size
            end_idx = min((group_idx + 1) * self.ortho_group_size, self.pop_size)

            group = np.arange(start_idx, end_idx)
            for j in range(self.dim):
                if np.random.rand() < self.CR or j == np.random.randint(0, self.dim):
                    crossed_population[i, j] = mutated_population[i, j]
                else:
                    crossed_population[i, j] = self.population[i, j]

        return crossed_population

    def handle_bounds(self, population, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        return np.clip(population, lb, ub)
    
    def stochastic_ranking(self, pop, fitness, func):
        N = len(pop)
        indices = np.arange(N)
        np.random.shuffle(indices)

        def constraint_violation(x):
            return 0

        def compare(i, j):
            fi = fitness[i] + constraint_violation(pop[i])
            fj = fitness[j] + constraint_violation(pop[j])
            
            return fi - fj
                
        ranked_indices = sorted(indices, key=lambda k: fitness[k] + constraint_violation(pop[k]))
        return pop[ranked_indices], fitness[ranked_indices]

    def crowding_distance(self, population, fitness):
        distances = np.zeros(len(population))
        for m in range(self.dim):
            dimension_values = population[:, m]
            sorted_indices = np.argsort(dimension_values)
            distances[sorted_indices[0]] = distances[sorted_indices[-1]] = np.inf

            for i in range(1, len(population) - 1):
                distances[sorted_indices[i]] += (dimension_values[sorted_indices[i+1]] - dimension_values[sorted_indices[i-1]])

        return distances

    def select(self, func, crossed_population):
        new_population = np.copy(self.population)
        new_fitness = np.copy(self.fitness)
        
        improved_count = 0

        for i in range(self.pop_size):
            f = func(crossed_population[i])
            self.budget -= 1

            if f < self.fitness[i]:
                new_population[i] = crossed_population[i]
                new_fitness[i] = f
                improved_count += 1
                
                self.successful_F.append(self.F)
                self.successful_CR.append(self.CR)

                if len(self.archive) < self.archive_size:
                    self.archive.append(self.population[i].copy())
                else:
                    archive_distances = self.crowding_distance(np.array(self.archive), np.zeros(len(self.archive)))
                    min_distance_idx = np.argmin(archive_distances)
                    self.archive[min_distance_idx] = self.population[i].copy()
        
        new_population, new_fitness = self.stochastic_ranking(new_population, new_fitness, func)

        if self.successful_F:
            mean_F = np.mean(self.successful_F)
            mean_CR = np.mean(self.successful_CR)

            self.F_momentum = self.momentum_coeff * self.F_momentum + (1 - self.momentum_coeff) * mean_F
            self.CR_momentum = self.momentum_coeff * self.CR_momentum + (1 - self.momentum_coeff) * mean_CR

            self.F = (1 - self.F_learning_rate) * self.F + self.F_learning_rate * self.F_momentum
            self.CR = (1 - self.CR_learning_rate) * self.CR + self.CR_learning_rate * self.CR_momentum

        self.successful_F = []
        self.successful_CR = []
        
        improvement_ratio = improved_count / self.pop_size
        if improvement_ratio > 0.3 and self.pop_size < self.pop_size_max:
            self.pop_size = min(self.pop_size + int(self.pop_size_adaptation_rate * (self.pop_size_max - self.pop_size)), self.pop_size_max)
        elif improvement_ratio < 0.1 and self.pop_size > self.pop_size_min:
            self.pop_size = max(self.pop_size - int(self.pop_size_adaptation_rate * (self.pop_size - self.pop_size_min)), self.pop_size_min)

        return new_population, new_fitness

    def check_stagnation(self):
        if len(self.best_fitness_history) >= self.stagnation_threshold:
            if np.std(self.best_fitness_history[-self.stagnation_threshold:]) < 1e-8:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
        
        if self.stagnation_counter >= 3:
            return True
        return False

    def restart(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        self.stagnation_counter = 0
        self.best_fitness_history = []

    def __call__(self, func):
        self.initialize_population(func)
        
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]
        self.best_fitness_history.append(self.f_opt)

        while self.budget > self.pop_size:
            mutated_population = self.mutate()
            crossed_population = self.crossover(mutated_population)
            crossed_population = self.handle_bounds(crossed_population, func)
            self.population, self.fitness = self.select(func, crossed_population)
            
            current_best_fitness = np.min(self.fitness)
            if current_best_fitness < self.f_opt:
                self.f_opt = current_best_fitness
                self.x_opt = self.population[np.argmin(self.fitness)]
            
            self.best_fitness_history.append(current_best_fitness)
            
            if self.check_stagnation():
                self.restart(func)
                self.best_fitness_history.append(np.min(self.fitness))
                
        return self.f_opt, self.x_opt