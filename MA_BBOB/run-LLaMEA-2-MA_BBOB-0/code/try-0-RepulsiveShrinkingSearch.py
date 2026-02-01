import numpy as np

class RepulsiveShrinkingSearch:
    def __init__(self, budget=10000, dim=10, pop_size=40, initial_radius=2.5, repulsion_factor=0.1, shrink_factor=0.995):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.initial_radius = initial_radius
        self.repulsion_factor = repulsion_factor
        self.shrink_factor = shrink_factor
        self.population = None
        self.fitness = None
        self.x_opt = None
        self.f_opt = np.inf
        self.current_radius = initial_radius

    def __call__(self, func):
        # Initialize population within the radius of the center
        center = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
        self.population = np.random.uniform(-self.current_radius, self.current_radius, size=(self.pop_size, self.dim)) + center
        self.population = np.clip(self.population, func.bounds.lb, func.bounds.ub)

        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size

        best_index = np.argmin(self.fitness)
        self.x_opt = self.population[best_index]
        self.f_opt = self.fitness[best_index]

        while self.budget > 0:
            # Repulsion Step
            for i in range(self.pop_size):
                repulsion_force = np.zeros(self.dim)
                for j in range(self.pop_size):
                    if i != j:
                        distance = np.linalg.norm(self.population[i] - self.population[j])
                        if distance > 0:
                            repulsion_direction = (self.population[i] - self.population[j]) / distance
                            repulsion_force += repulsion_direction / distance  # Inverse distance repulsion

                # Move the individual based on the repulsion force
                new_position = self.population[i] + self.repulsion_factor * repulsion_force
                new_position = np.clip(new_position, func.bounds.lb, func.bounds.ub)

                f_new = func(new_position)
                self.budget -= 1

                if f_new < self.f_opt:
                    self.f_opt = f_new
                    self.x_opt = new_position

                if f_new < self.fitness[i]:
                    self.fitness[i] = f_new
                    self.population[i] = new_position

            # Shrink the search radius
            self.current_radius *= self.shrink_factor
            
            # Move population to within the radius
            center = self.population[np.argmin(self.fitness)]
            self.population = np.random.uniform(-self.current_radius, self.current_radius, size=(self.pop_size, self.dim)) + center
            self.population = np.clip(self.population, func.bounds.lb, func.bounds.ub)
            self.fitness = np.array([func(x) for x in self.population])
            self.budget -= self.pop_size
            
            best_index = np.argmin(self.fitness)
            if self.fitness[best_index] < self.f_opt:
                self.f_opt = self.fitness[best_index]
                self.x_opt = self.population[best_index]

        return self.f_opt, self.x_opt