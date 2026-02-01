import numpy as np

class SelfOrganizingPSO:
    def __init__(self, budget=10000, dim=10, pop_size=40, inertia=0.7, c1=1.5, c2=1.5, velocity_clamp=0.5):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.inertia = inertia
        self.c1 = c1
        self.c2 = c2
        self.velocity_clamp = velocity_clamp
        self.swarm = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_fitness = None
        self.global_best_position = None
        self.global_best_fitness = np.inf

    def initialize_swarm(self, func):
        self.swarm = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.velocities = np.random.uniform(-self.velocity_clamp, self.velocity_clamp, size=(self.pop_size, self.dim))
        self.personal_best_positions = np.copy(self.swarm)
        self.personal_best_fitness = np.array([func(x) for x in self.swarm])
        self.budget -= self.pop_size

        self.global_best_position = self.personal_best_positions[np.argmin(self.personal_best_fitness)]
        self.global_best_fitness = np.min(self.personal_best_fitness)
        self.x_opt = self.global_best_position
        self.f_opt = self.global_best_fitness

    def update_velocities(self):
        r1 = np.random.rand(self.pop_size, self.dim)
        r2 = np.random.rand(self.pop_size, self.dim)
        cognitive_component = self.c1 * r1 * (self.personal_best_positions - self.swarm)
        social_component = self.c2 * r2 * (self.global_best_position - self.swarm)
        self.velocities = self.inertia * self.velocities + cognitive_component + social_component
        self.velocities = np.clip(self.velocities, -self.velocity_clamp, self.velocity_clamp)

    def update_positions(self, func):
        self.swarm += self.velocities
        self.swarm = np.clip(self.swarm, func.bounds.lb, func.bounds.ub)

        fitness = np.array([func(x) for x in self.swarm])
        self.budget -= self.pop_size

        for i in range(self.pop_size):
            if fitness[i] < self.personal_best_fitness[i]:
                self.personal_best_fitness[i] = fitness[i]
                self.personal_best_positions[i] = self.swarm[i]

                if fitness[i] < self.global_best_fitness:
                    self.global_best_fitness = fitness[i]
                    self.global_best_position = self.swarm[i]
                    self.x_opt = self.global_best_position
                    self.f_opt = self.global_best_fitness

    def calculate_diversity(self):
        centroid = np.mean(self.swarm, axis=0)
        distances = np.linalg.norm(self.swarm - centroid, axis=1)
        diversity = np.mean(distances)
        return diversity

    def adjust_parameters(self, diversity):
        # Dynamically adjust inertia and exploration/exploitation balance
        if diversity > 0.1 * (func.bounds.ub[0] - func.bounds.lb[0]):  # Arbitrary threshold
            self.inertia = min(0.9, self.inertia + 0.05)  # Increase inertia for exploration
            self.c1 = max(1.0, self.c1 - 0.05)          # Reduce cognitive component
            self.c2 = min(2.0, self.c2 + 0.05)          # Increase social component
        else:
            self.inertia = max(0.4, self.inertia - 0.05)  # Decrease inertia for exploitation
            self.c1 = min(2.0, self.c1 + 0.05)          # Increase cognitive component
            self.c2 = max(1.0, self.c2 - 0.05)          # Reduce social component

    def __call__(self, func):
        self.initialize_swarm(func)

        while self.budget > 0:
            diversity = self.calculate_diversity()
            self.adjust_parameters(diversity)
            n_evaluations_before = self.budget
            self.update_velocities()
            self.update_positions(func)
            n_evaluations_after = self.budget

            if n_evaluations_after == n_evaluations_before:
              break # no more budget left, so stop optimization

        return self.f_opt, self.x_opt