import numpy as np

class SelfOrganizingMigratingAlgorithm:
    def __init__(self, budget=10000, dim=10, pop_size=50, migration_probability=0.1, initial_step_size=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.migration_probability = migration_probability
        self.initial_step_size = initial_step_size
        self.population = None
        self.fitness = None
        self.step_size = None
        self.f_opt = np.Inf
        self.x_opt = None
        self.eval_count = 0
        self.success_rate = 0.0
        self.success_history = []
        self.success_history_size = 10

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.eval_count += self.pop_size
        self.step_size = np.full(self.pop_size, self.initial_step_size)
        
        best_index = np.argmin(self.fitness)
        if self.fitness[best_index] < self.f_opt:
            self.f_opt = self.fitness[best_index]
            self.x_opt = self.population[best_index].copy()

    def migrate(self, func):
        for i in range(self.pop_size):
            # Select a neighbor - probabilistic selection based on fitness
            probabilities = np.exp(-self.fitness / self.f_opt)  # Higher fitness -> lower probability
            probabilities /= np.sum(probabilities)
            
            neighbors = np.arange(self.pop_size)
            neighbor_index = np.random.choice(neighbors, p=probabilities)

            if np.random.rand() < self.migration_probability:
                # Adaptive step size
                mutation = self.step_size[i] * np.random.normal(0, 1, self.dim)
                x_new = self.population[i] + mutation
                x_new = np.clip(x_new, func.bounds.lb, func.bounds.ub)
                
                f_new = func(x_new)
                self.eval_count += 1

                if f_new < self.fitness[i]:
                    self.success_history.append(1)
                    self.population[i] = x_new
                    self.fitness[i] = f_new
                    
                    if f_new < self.f_opt:
                        self.f_opt = f_new
                        self.x_opt = x_new.copy()
                    self.step_size[i] *= 1.1 # Increase step size if successful
                else:
                    self.success_history.append(0)
                    self.step_size[i] *= 0.9 # Decrease step size if unsuccessful

                # Keep track of success rate for global step size adjustment
                if len(self.success_history) > self.success_history_size:
                    self.success_history = self.success_history[-self.success_history_size:]

                self.success_rate = np.mean(self.success_history)
                
                # Global step size adaptation based on success rate.
                if self.success_rate > 0.6:
                     self.step_size *= 1.05
                elif self.success_rate < 0.2:
                    self.step_size *= 0.95

            if self.eval_count >= self.budget:
                break

    def __call__(self, func):
        self.initialize_population(func)
        while self.eval_count < self.budget:
            self.migrate(func)
        return self.f_opt, self.x_opt