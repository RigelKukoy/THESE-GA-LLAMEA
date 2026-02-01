import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10, num_particles=20, inertia_max=0.9, inertia_min=0.2, c1=2.0, c2=2.0, v_max_ratio=0.2):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.inertia_max = inertia_max
        self.inertia_min = inertia_min
        self.c1 = c1
        self.c2 = c2
        self.lb = -5.0
        self.ub = 5.0
        self.v_max = v_max_ratio * (self.ub - self.lb)  # Velocity clamping
        self.particles = np.random.uniform(self.lb, self.ub, size=(self.num_particles, self.dim))
        self.velocities = np.random.uniform(-self.v_max, self.v_max, size=(self.num_particles, self.dim))  # Initialize velocities
        self.personal_best_positions = self.particles.copy()
        self.personal_best_values = np.full(self.num_particles, np.inf)
        self.global_best_position = None
        self.global_best_value = np.inf
        self.eval_count = 0

    def __call__(self, func):
        while self.eval_count < self.budget:
            # Evaluate particles
            fitness_values = np.zeros(self.num_particles)
            for i in range(self.num_particles):
                if self.eval_count < self.budget:
                    fitness_values[i] = func(self.particles[i])
                    self.eval_count += 1
                else:
                    fitness_values[i] = np.inf  # Or a very large number

            # Update personal bests
            for i in range(self.num_particles):
                if fitness_values[i] < self.personal_best_values[i]:
                    self.personal_best_values[i] = fitness_values[i]
                    self.personal_best_positions[i] = self.particles[i].copy()

            # Update global best
            best_index = np.argmin(fitness_values)
            if fitness_values[best_index] < self.global_best_value:
                self.global_best_value = fitness_values[best_index]
                self.global_best_position = self.particles[best_index].copy()

            # Update inertia weight (linearly decreasing)
            inertia = self.inertia_max - (self.inertia_max - self.inertia_min) * (self.eval_count / self.budget)

            # Update velocities and positions
            for i in range(self.num_particles):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                self.velocities[i] = (inertia * self.velocities[i] +
                                      self.c1 * r1 * (self.personal_best_positions[i] - self.particles[i]) +
                                      self.c2 * r2 * (self.global_best_position - self.particles[i]))

                # Velocity clamping
                self.velocities[i] = np.clip(self.velocities[i], -self.v_max, self.v_max)
                self.particles[i] = self.particles[i] + self.velocities[i]

                # Boundary handling (clip or bounce)
                self.particles[i] = np.clip(self.particles[i], self.lb, self.ub)

        return self.global_best_value, self.global_best_position