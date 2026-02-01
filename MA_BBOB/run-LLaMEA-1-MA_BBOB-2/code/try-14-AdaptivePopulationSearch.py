import numpy as np

class AdaptivePopulationSearch:
    def __init__(self, budget=10000, dim=10, pop_size=20, exploration_rate=0.5, top_k=5):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.exploration_rate = exploration_rate
        self.lb = -5.0
        self.ub = 5.0
        self.top_k = top_k  # Number of top individuals to consider for exploitation
        self.step_size = 0.1 * (self.ub - self.lb)  # Initial step size

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        self.budget -= self.pop_size

        # Find best individual
        best_index = np.argmin(fitness)
        self.f_opt = fitness[best_index]
        self.x_opt = population[best_index]

        # Main loop
        success_count = 0
        while self.budget > 0:
            # Adaptive adjustment of exploration rate based on fitness variance
            fitness_variance = np.var(fitness)
            if fitness_variance > 1e-3: # If variance is high, explore more
                exploration_rate = min(self.exploration_rate + 0.1, 0.9)
            else: # If variance is low, exploit more
                exploration_rate = max(self.exploration_rate - 0.1, 0.1)

            new_population = np.copy(population)
            for i in range(self.pop_size):
                if np.random.rand() < exploration_rate:
                    # Exploration: Randomly perturb the individual
                    new_population[i] = population[i] + np.random.uniform(-1.0, 1.0, size=self.dim) * self.step_size
                    new_population[i] = np.clip(new_population[i], self.lb, self.ub)
                else:
                    # Exploitation: Guided by the top k individuals
                    top_indices = np.argsort(fitness)[:self.top_k]
                    selected_top = population[np.random.choice(top_indices)] # select one of the top k individuals
                    new_population[i] = population[i] + np.random.uniform(0, 1, size=self.dim) * (selected_top - population[i])
                    new_population[i] = np.clip(new_population[i], self.lb, self.ub)


            new_fitness = np.array([func(x) for x in new_population])
            self.budget -= self.pop_size
            
            # Update population (replace if better)
            for i in range(self.pop_size):
                if new_fitness[i] < fitness[i]:
                    success_count += 1
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]
                    
                    if new_fitness[i] < self.f_opt:
                        self.f_opt = new_fitness[i]
                        self.x_opt = new_population[i]

            # Adjust step size based on success rate
            if success_count > 0.2 * self.pop_size:
                self.step_size *= 1.1  # Increase step size if many improvements
                success_count = 0
            else:
                self.step_size *= 0.9  # Decrease step size if few improvements
            self.step_size = np.clip(self.step_size, 0.01 * (self.ub - self.lb), 0.5 * (self.ub - self.lb))


        return self.f_opt, self.x_opt