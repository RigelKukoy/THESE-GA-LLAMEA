import numpy as np

class SelfOrganizingSearch:
    def __init__(self, budget=10000, dim=10, pop_size=40, initial_step_size=0.1, step_size_decay=0.99, success_memory=10, repulsion_factor=0.01):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.initial_step_size = initial_step_size
        self.step_size = np.full(pop_size, initial_step_size)
        self.step_size_decay = step_size_decay
        self.success_memory = success_memory
        self.success_history = np.zeros((pop_size, success_memory))
        self.repulsion_factor = repulsion_factor
        self.population = None
        self.fitness = None
        self.x_opt = None
        self.f_opt = np.inf

    def __call__(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size

        best_index = np.argmin(self.fitness)
        self.x_opt = self.population[best_index]
        self.f_opt = self.fitness[best_index]

        while self.budget > 0:
            for i in range(self.pop_size):
                # Adaptive step size adjustment
                success_rate = np.mean(self.success_history[i])
                self.step_size[i] = self.initial_step_size * (self.step_size_decay ** (1 - success_rate))

                # Generate a candidate solution
                mutation = np.random.normal(0, self.step_size[i], size=self.dim)
                new_position = self.population[i] + mutation
                new_position = np.clip(new_position, func.bounds.lb, func.bounds.ub)

                # Repulsion from other individuals to maintain diversity
                for j in range(self.pop_size):
                    if i != j:
                        direction = self.population[i] - self.population[j]
                        distance = np.linalg.norm(direction)
                        if distance > 0:
                            new_position += self.repulsion_factor * direction / (distance + 1e-8)
                            new_position = np.clip(new_position, func.bounds.lb, func.bounds.ub)

                f_new = func(new_position)
                self.budget -= 1

                # Update individual's position and success history
                if f_new < self.f_opt:
                    self.f_opt = f_new
                    self.x_opt = new_position

                if f_new < self.fitness[i]:
                    self.fitness[i] = f_new
                    self.population[i] = new_position
                    self.success_history[i] = np.roll(self.success_history[i], 1)
                    self.success_history[i][0] = 1
                else:
                    self.success_history[i] = np.roll(self.success_history[i], 1)
                    self.success_history[i][0] = 0

            best_index = np.argmin(self.fitness)
            if self.fitness[best_index] < self.f_opt:
                self.f_opt = self.fitness[best_index]
                self.x_opt = self.population[best_index]

        return self.f_opt, self.x_opt