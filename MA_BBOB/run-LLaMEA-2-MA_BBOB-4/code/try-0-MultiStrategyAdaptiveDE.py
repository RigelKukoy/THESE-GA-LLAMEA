import numpy as np

class MultiStrategyAdaptiveDE:
    def __init__(self, budget=10000, dim=10, pop_size=40, Cr=0.9, F_initial=0.5, strategy_prob=[0.3, 0.3, 0.4], diversity_threshold=0.1, success_memory=10):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.Cr = Cr
        self.F = F_initial
        self.strategy_prob = strategy_prob  # Probabilities for each mutation strategy
        self.diversity_threshold = diversity_threshold
        self.success_memory = success_memory
        self.success_rates = [0.0] * len(self.strategy_prob)  # Track success rates of each strategy
        self.strategy_counts = [0] * len(self.strategy_prob)
        self.best_fitness_history = []
        self.last_improvement = 0
        self.generation = 0

    def calculate_diversity(self, population):
        """Calculates the diversity of the population based on the mean pairwise distance."""
        distances = np.sum((population[:, np.newaxis, :] - population[np.newaxis, :, :]) ** 2, axis=2)
        diversity = np.mean(distances)
        return diversity

    def mutation_strategy(self, population, i, strategy_index):
        """Applies different mutation strategies based on the strategy index."""
        if strategy_index == 0:  # DE/rand/1
            indices = np.random.choice(self.pop_size, 3, replace=False)
            x_r1, x_r2, x_r3 = population[indices]
            return x_r1 + self.F * (x_r2 - x_r3)
        elif strategy_index == 1:  # DE/current-to-rand/1
            indices = np.random.choice(self.pop_size, 2, replace=False)
            x_r1, x_r2 = population[indices]
            return population[i] + self.F * (x_r1 - population[i]) + self.F * (x_r2 - population[i])
        elif strategy_index == 2:  # DE/current-to-best/1
            best_index = np.argmin(self.fitness)
            indices = np.random.choice(self.pop_size, 2, replace=False)
            x_r1, x_r2 = population[indices]
            return population[i] + self.F * (population[best_index] - population[i]) + self.F * (x_r1 - x_r2)
        else:
            raise ValueError("Invalid strategy index.")

    def __call__(self, func):
        # Initialization
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size
        
        self.f_opt = np.min(self.fitness)
        self.x_opt = population[np.argmin(self.fitness)]
        self.best_fitness_history.append(self.f_opt)
        self.last_improvement = 0
        self.generation = 0
        success_memory = []

        while self.budget > self.pop_size:
            # Dynamic adjustment of strategy probabilities based on success
            if len(success_memory) >= self.success_memory:
                success_memory.pop(0)

            new_population = np.copy(population)
            new_fitness = np.copy(self.fitness)

            for i in range(self.pop_size):
                # Select mutation strategy based on probabilities
                strategy_index = np.random.choice(len(self.strategy_prob), p=self.strategy_prob)
                self.strategy_counts[strategy_index] += 1

                # Mutation
                mutant = self.mutation_strategy(population, i, strategy_index)

                # Crossover
                for j in range(self.dim):
                    if np.random.rand() < self.Cr:
                        new_population[i, j] = mutant[j]
                    else:
                        new_population[i, j] = population[i, j]
                
                new_population[i] = np.clip(new_population[i], func.bounds.lb, func.bounds.ub)

            # Evaluation
            new_fitness = np.array([func(x) for x in new_population])
            self.budget -= self.pop_size

            # Selection and Success tracking
            for i in range(self.pop_size):
                if new_fitness[i] < self.fitness[i]:
                    # Strategy success
                    success_memory.append(strategy_index)
                    population[i] = new_population[i]
                    self.fitness[i] = new_fitness[i]

                    if new_fitness[i] < self.f_opt:
                        self.f_opt = new_fitness[i]
                        self.x_opt = new_population[i]
                        self.last_improvement = self.generation

            self.best_fitness_history.append(self.f_opt)

            # Calculate and adjust strategy probabilities based on recent successes
            success_counts = [success_memory.count(k) for k in range(len(self.strategy_prob))]
            total_successes = sum(success_counts)
            if total_successes > 0:
                self.strategy_prob = [(count / total_successes) for count in success_counts]
            else: # If there are no success, keep initial probabilities.
                self.strategy_prob = [p for p in self.strategy_prob]
                self.strategy_prob = [p / sum(self.strategy_prob) for p in self.strategy_prob] # Normalize

            # Diversity check
            diversity = self.calculate_diversity(population)
            if diversity < self.diversity_threshold:
                # Increase exploration by increasing mutation rate
                self.F = min(self.F * 1.1, 1.0)
            else:
                # Decrease exploration if diversity is high
                self.F = max(self.F * 0.9, 0.1)
            
            self.generation += 1

        return self.f_opt, self.x_opt