import numpy as np

class AdaptiveDE_OrthoEnhanced:
    def __init__(self, budget=10000, dim=10, pop_size_min=20, pop_size_max=100, archive_size=10, F_initial=0.5, CR_initial=0.5, F_learning_rate=0.1, CR_learning_rate=0.1, stagnation_threshold=500, ortho_group_size=5, local_search_probability=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size_min = pop_size_min
        self.pop_size_max = pop_size_max
        self.pop_size = pop_size_max # Start with a larger population
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
        self.stagnation_threshold = stagnation_threshold
        self.stagnation_counter = 0
        self.best_fitness_history = []
        self.ortho_group_size = ortho_group_size # Size of groups for orthogonal crossover
        self.pop_size_adaptation_rate = 0.1 # Rate to adjust pop size
        self.local_search_probability = local_search_probability
        self.age = np.zeros(pop_size_max)
        self.age_limit = 50


    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        self.archive = []
        self.age = np.zeros(self.pop_size)

    def mutate(self):
        mutated_population = np.zeros_like(self.population)
        for i in range(self.pop_size):
            if np.random.rand() < 0.5: # Current-to-best mutation
                best_idx = np.argmin(self.fitness)
                idxs = np.random.choice(self.pop_size, 2, replace=False)
                x1, x2 = self.population[idxs]
                mutated_population[i] = self.population[i] + self.F * (self.population[best_idx] - self.population[i]) + self.F * (x1 - x2)
            else: # Random mutation
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                x1, x2, x3 = self.population[idxs]
                mutated_population[i] = x1 + self.F * (x2 - x3)
        return mutated_population

    def crossover(self, mutated_population):
        crossed_population = np.zeros_like(self.population)
        for i in range(self.pop_size):
            # Perform orthogonal crossover in groups
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
                self.age[i] = 0 # Reset age

                self.successful_F.append(self.F)
                self.successful_CR.append(self.CR)
            else:
                self.age[i] += 1

        # Aging: Replace old individuals
        for i in range(self.pop_size):
            if self.age[i] > self.age_limit:
                new_population[i] = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
                new_fitness[i] = func(new_population[i])
                self.budget -=1
                self.age[i] = 0
                
        # Local search refinement
        for i in range(self.pop_size):
            if np.random.rand() < self.local_search_probability:
                step_size = 0.01 * (func.bounds.ub - func.bounds.lb)
                new_x = new_population[i] + np.random.uniform(-step_size, step_size, size=self.dim)
                new_x = self.handle_bounds(new_x, func)
                new_f = func(new_x)
                self.budget -= 1
                if new_f < new_fitness[i]:
                    new_population[i] = new_x
                    new_fitness[i] = new_f
                    
        # Population size adaptation
        improvement_ratio = improved_count / self.pop_size
        if improvement_ratio > 0.3 and self.pop_size < self.pop_size_max:
            self.pop_size = min(self.pop_size + int(self.pop_size_adaptation_rate * (self.pop_size_max - self.pop_size)), self.pop_size_max)
            self.population = np.concatenate([new_population, np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size - len(new_population), self.dim))])
            self.fitness = np.concatenate([new_fitness, np.array([func(x) for x in self.population[len(new_population):]])])
            self.age = np.concatenate([self.age, np.zeros(self.pop_size - len(new_population))])
            self.budget -= (self.pop_size - len(new_population))
        elif improvement_ratio < 0.1 and self.pop_size > self.pop_size_min:
            self.pop_size = max(self.pop_size - int(self.pop_size_adaptation_rate * (self.pop_size - self.pop_size_min)), self.pop_size_min)
            # Keep the best and reduce population
            sorted_indices = np.argsort(new_fitness)[:self.pop_size]
            self.population = new_population[sorted_indices]
            self.fitness = new_fitness[sorted_indices]
            self.age = self.age[sorted_indices]


        if self.successful_F:
            self.F = np.mean(self.successful_F)
            self.CR = np.mean(self.successful_CR)

        self.successful_F = []
        self.successful_CR = []

        return new_population, new_fitness

    def check_stagnation(self):
        N = min(len(self.best_fitness_history), self.stagnation_threshold)
        if N < self.stagnation_threshold:
            return False

        if np.std(self.best_fitness_history[-N:]) < 1e-8:
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
        self.age = np.zeros(self.pop_size) # Reset age
        self.best_fitness_history = []

    def __call__(self, func):
        self.initialize_population(func)
        
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)]
        self.best_fitness_history.append(self.f_opt)

        while self.budget > self.pop_size_min: # Ensure enough budget for minimal population updates
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