import numpy as np

class AdaptiveLandscapeExplorer:
    def __init__(self, budget=10000, dim=10, pop_size=50, exploration_prob=0.3, local_search_intensity=0.1, step_size_init=0.2):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.exploration_prob = exploration_prob
        self.local_search_intensity = local_search_intensity
        self.step_size = step_size_init
        self.population = None
        self.fitness = None
        self.best_x = None
        self.best_f = np.inf

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub

        # Initialize population
        self.population = np.random.uniform(lb, ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size

        # Find initial best solution
        best_idx = np.argmin(self.fitness)
        self.best_x = self.population[best_idx]
        self.best_f = self.fitness[best_idx]

        while self.budget > 0:
            new_population = np.copy(self.population)
            new_fitness = np.copy(self.fitness)

            for i in range(self.pop_size):
                if np.random.rand() < self.exploration_prob:
                    # Global exploration: Randomly sample a new solution
                    new_x = np.random.uniform(lb, ub, size=self.dim)
                else:
                    # Local refinement: Perturb the current solution
                    new_x = self.population[i] + np.random.uniform(-self.step_size, self.step_size, size=self.dim) * self.local_search_intensity
                    new_x = np.clip(new_x, lb, ub)

                new_f = func(new_x)
                self.budget -= 1
                
                if new_f < self.best_f:
                    self.best_f = new_f
                    self.best_x = new_x

                if new_f < self.fitness[i]:
                    new_population[i] = new_x
                    new_fitness[i] = new_f
            
            # Update population and fitness
            self.population = new_population
            self.fitness = new_fitness

            # Dynamically adjust step size (exploration vs. exploitation)
            improvement_ratio = np.sum(self.fitness - self.population) / self.pop_size
            if improvement_ratio > 0:
              self.step_size *= 0.95 # Reduce step size
            else:
              self.step_size *= 1.05 # Increase step size
            self.step_size = np.clip(self.step_size, 0.01, 0.5)

        return self.best_f, self.best_x