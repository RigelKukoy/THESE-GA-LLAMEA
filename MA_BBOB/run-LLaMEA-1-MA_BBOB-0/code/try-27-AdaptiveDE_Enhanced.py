import numpy as np

class AdaptiveDE_Enhanced:
    def __init__(self, budget=10000, dim=10, pop_size=50, archive_size=10, F_initial=0.5, CR_initial=0.5, F_learning_rate=0.1, CR_learning_rate=0.1, stagnation_threshold=1000):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.archive_size = archive_size
        self.F = F_initial  # Mutation factor
        self.CR = CR_initial  # Crossover rate
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

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        self.archive = []

    def mutate(self):
        mutated_population = np.zeros_like(self.population)
        for i in range(self.pop_size):
            idxs = np.random.choice(self.pop_size, 3, replace=False)
            x1, x2, x3 = self.population[idxs]
            mutated_population[i] = x1 + self.F * (x2 - x3)
            
        return mutated_population

    def crossover(self, mutated_population):
        crossed_population = np.zeros_like(self.population)
        for i in range(self.pop_size):
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
    
    def stochastic_ranking(self, pop, fitness):
        N = len(pop)
        indices = np.arange(N)
        np.random.shuffle(indices)
        
        def compare(i, j):
            fi = fitness[i]
            fj = fitness[j]
            
            if (fi < 0 and fj < 0) or (fi >= 0 and fj >= 0):
                return fi - fj
            elif fi < 0 and fj >= 0:
                return -1
            else:
                return 1
                
        ranked_indices = sorted(indices, key=lambda k: fitness[k])
        return pop[ranked_indices], fitness[ranked_indices]

    def crowding_distance(self, population, fitness):
        distances = np.zeros(len(population))
        for m in range(self.dim):  # Consider each dimension
            dimension_values = population[:, m]
            sorted_indices = np.argsort(dimension_values)
            distances[sorted_indices[0]] = distances[sorted_indices[-1]] = np.inf  # Boundary individuals

            for i in range(1, len(population) - 1):
                distances[sorted_indices[i]] += (dimension_values[sorted_indices[i+1]] - dimension_values[sorted_indices[i-1]])

        return distances

    def select(self, func, crossed_population):
        new_population = np.copy(self.population)
        new_fitness = np.copy(self.fitness)
        
        for i in range(self.pop_size):
            f = func(crossed_population[i])
            self.budget -= 1

            if f < self.fitness[i]:
                new_population[i] = crossed_population[i]
                new_fitness[i] = f
                
                self.successful_F.append(self.F)
                self.successful_CR.append(self.CR)

                # Archive update using crowding distance
                if len(self.archive) < self.archive_size:
                    self.archive.append(self.population[i].copy())
                else:
                    archive_distances = self.crowding_distance(np.array(self.archive), np.zeros(len(self.archive))) # Dummy fitness values for crowding distance
                    min_distance_idx = np.argmin(archive_distances)
                    self.archive[min_distance_idx] = self.population[i].copy() # Replace least crowded
        
        new_population, new_fitness = self.stochastic_ranking(new_population, new_fitness)

        # Update F and CR with momentum
        if self.successful_F:
            mean_F = np.mean(self.successful_F)
            mean_CR = np.mean(self.successful_CR)

            self.F_momentum = self.momentum_coeff * self.F_momentum + (1 - self.momentum_coeff) * mean_F
            self.CR_momentum = self.momentum_coeff * self.CR_momentum + (1 - self.momentum_coeff) * mean_CR

            self.F = (1 - self.F_learning_rate) * self.F + self.F_learning_rate * self.F_momentum
            self.CR = (1 - self.CR_learning_rate) * self.CR + self.CR_learning_rate * self.CR_momentum

        self.successful_F = []
        self.successful_CR = []

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
        self.stagnation_counter = 0  # Reset stagnation counter
        self.best_fitness_history = []

    def __call__(self, func):
        self.initialize_population(func)
        
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]
        self.best_fitness_history.append(self.f_opt)

        while self.budget > 0:
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
                self.best_fitness_history.append(np.min(self.fitness))  # Record new best fitness after restart
                
        return self.f_opt, self.x_opt