import numpy as np

class SelfOrganizingScoutPSO:
    def __init__(self, budget=10000, dim=10, pop_size_min=20, pop_size_max=100, scout_rate=0.1, inertia=0.7, cognitive_coeff=1.4, social_coeff=1.4, velocity_clamp=0.5):
        self.budget = budget
        self.dim = dim
        self.pop_size_min = pop_size_min
        self.pop_size_max = pop_size_max
        self.pop_size = pop_size_max #start with a bigger population and decrease it if needed
        self.scout_rate = scout_rate
        self.inertia = inertia
        self.cognitive_coeff = cognitive_coeff
        self.social_coeff = social_coeff
        self.velocity_clamp = velocity_clamp
        self.f_opt = np.Inf
        self.x_opt = None
        self.swarm = None
        self.velocities = None
        self.local_best_positions = None
        self.local_best_fitness = None
        self.global_best_position = None
        self.global_best_fitness = np.Inf

    def initialize_swarm(self, func):
        self.swarm = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.velocities = np.random.uniform(-self.velocity_clamp, self.velocity_clamp, size=(self.pop_size, self.dim))
        self.local_best_positions = self.swarm.copy()
        self.local_best_fitness = np.array([func(x) for x in self.swarm])
        self.budget -= self.pop_size

        self.global_best_position = self.local_best_positions[np.argmin(self.local_best_fitness)].copy()
        self.global_best_fitness = np.min(self.local_best_fitness)

        if self.global_best_fitness < self.f_opt:
            self.f_opt = self.global_best_fitness
            self.x_opt = self.global_best_position.copy()

    def update_velocity(self, i):
        r1 = np.random.rand(self.dim)
        r2 = np.random.rand(self.dim)
        cognitive_component = self.cognitive_coeff * r1 * (self.local_best_positions[i] - self.swarm[i])
        social_component = self.social_coeff * r2 * (self.global_best_position - self.swarm[i])
        self.velocities[i] = self.inertia * self.velocities[i] + cognitive_component + social_component
        self.velocities[i] = np.clip(self.velocities[i], -self.velocity_clamp, self.velocity_clamp)

    def update_position(self, i, func):
        new_position = self.swarm[i] + self.velocities[i]
        new_position = np.clip(new_position, func.bounds.lb, func.bounds.ub)
        return new_position

    def scout(self, func):
        # Replace a percentage of worst performing particles with new random ones
        num_scouts = int(self.scout_rate * self.pop_size)
        worst_indices = np.argsort(self.local_best_fitness)[-num_scouts:] # Indices of worst particles

        for i in worst_indices:
            self.swarm[i] = np.random.uniform(func.bounds.lb, func.bounds.ub)
            self.velocities[i] = np.random.uniform(-self.velocity_clamp, self.velocity_clamp, self.dim)
            fitness = func(self.swarm[i])
            self.budget -= 1

            if fitness < self.f_opt:
                self.f_opt = fitness
                self.x_opt = self.swarm[i].copy()

            self.local_best_positions[i] = self.swarm[i].copy()
            self.local_best_fitness[i] = fitness

            if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = self.swarm[i].copy()


    def adjust_population_size(self):
        # Dynamically adjust population size based on stagnation
        if np.std(self.local_best_fitness) < 1e-6 and self.pop_size > self.pop_size_min:
            self.pop_size = max(self.pop_size_min, int(self.pop_size * 0.9))  # Reduce population size
            self.swarm = self.swarm[:self.pop_size]
            self.velocities = self.velocities[:self.pop_size]
            self.local_best_positions = self.local_best_positions[:self.pop_size]
            self.local_best_fitness = self.local_best_fitness[:self.pop_size]
        elif np.std(self.local_best_fitness) > 0.1 and self.pop_size < self.pop_size_max:
            self.pop_size = min(self.pop_size_max, int(self.pop_size * 1.1))

    def __call__(self, func):
        self.initialize_swarm(func)

        while self.budget > 0:
            for i in range(self.pop_size):
                self.update_velocity(i)
                new_position = self.update_position(i, func)
                
                fitness = func(new_position)
                self.budget -= 1

                if fitness < self.f_opt:
                    self.f_opt = fitness
                    self.x_opt = new_position.copy()

                if fitness < self.local_best_fitness[i]:
                    self.local_best_fitness[i] = fitness
                    self.local_best_positions[i] = new_position.copy()

                    if fitness < self.global_best_fitness:
                        self.global_best_fitness = fitness
                        self.global_best_position = new_position.copy()
                        
            self.scout(func) # Scout for new regions
            self.adjust_population_size()

        return self.f_opt, self.x_opt