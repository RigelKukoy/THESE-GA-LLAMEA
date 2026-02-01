import numpy as np

class CooperativeSwarm:
    def __init__(self, budget=10000, dim=10, num_swarms=5, swarm_size=10, radius_initial=1.0, radius_final=0.1, inertia=0.7, cognitive_coeff=1.4, social_coeff=1.4):
        self.budget = budget
        self.dim = dim
        self.num_swarms = num_swarms
        self.swarm_size = swarm_size
        self.radius_initial = radius_initial
        self.radius_final = radius_final
        self.inertia = inertia
        self.cognitive_coeff = cognitive_coeff
        self.social_coeff = social_coeff
        self.swarms = None
        self.swarm_fitness = None
        self.swarm_best_positions = None
        self.swarm_best_fitness = None
        self.global_best_position = None
        self.global_best_fitness = np.inf
        self.eval_count = 0
        self.velocities = None

    def initialize_swarms(self, func):
        self.swarms = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.num_swarms, self.swarm_size, self.dim))
        self.velocities = np.zeros_like(self.swarms)
        self.swarm_fitness = np.zeros((self.num_swarms, self.swarm_size))
        self.swarm_best_positions = np.copy(self.swarms)
        self.swarm_best_fitness = np.full((self.num_swarms, self.swarm_size), np.inf)

        for i in range(self.num_swarms):
            for j in range(self.swarm_size):
                self.swarm_fitness[i, j] = func(self.swarms[i, j])
                self.eval_count += 1
                self.swarm_best_fitness[i, j] = self.swarm_fitness[i, j]
                if self.swarm_fitness[i, j] < self.global_best_fitness:
                    self.global_best_fitness = self.swarm_fitness[i, j]
                    self.global_best_position = np.copy(self.swarms[i, j])

    def __call__(self, func):
        self.initialize_swarms(func)

        while self.eval_count < self.budget:
            # Adaptive Radius
            remaining_evals = self.budget - self.eval_count
            progress = 1.0 - (remaining_evals / self.budget)
            radius = self.radius_initial + (self.radius_final - self.radius_initial) * progress

            for i in range(self.num_swarms):
                # Find local best in the swarm
                local_best_index = np.argmin(self.swarm_best_fitness[i])
                local_best_position = self.swarm_best_positions[i, local_best_index]

                for j in range(self.swarm_size):
                    # Update velocity
                    r1 = np.random.rand(self.dim)
                    r2 = np.random.rand(self.dim)
                    self.velocities[i, j] = (self.inertia * self.velocities[i, j] +
                                            self.cognitive_coeff * r1 * (self.swarm_best_positions[i, j] - self.swarms[i, j]) +
                                            self.social_coeff * r2 * (local_best_position - self.swarms[i, j]))

                    # Update position with radius-based exploration
                    self.swarms[i, j] = self.swarms[i, j] + self.velocities[i, j]
                    
                    # Radius-based exploration: occasionally explore within a shrinking radius
                    if np.random.rand() < 0.1:  # Probability of exploration
                        exploration_vector = np.random.uniform(-radius, radius, size=self.dim)
                        self.swarms[i, j] = self.swarm_best_positions[i, j] + exploration_vector
                    
                    self.swarms[i, j] = np.clip(self.swarms[i, j], func.bounds.lb, func.bounds.ub)


                    # Evaluate new position
                    fitness = func(self.swarms[i, j])
                    self.eval_count += 1

                    # Update personal best
                    if fitness < self.swarm_best_fitness[i, j]:
                        self.swarm_best_fitness[i, j] = fitness
                        self.swarm_best_positions[i, j] = np.copy(self.swarms[i, j])

                    # Update global best
                    if fitness < self.global_best_fitness:
                        self.global_best_fitness = fitness
                        self.global_best_position = np.copy(self.swarms[i, j])
                    
                    if self.eval_count >= self.budget:
                        break
                if self.eval_count >= self.budget:
                    break

            # Dynamic sub-swarm merging (optional, but might help)
            if self.num_swarms > 1 and self.eval_count % (self.budget // 10) == 0:
                # Periodically merge the two closest swarms based on distance of their best particles
                swarm_distances = np.zeros((self.num_swarms, self.num_swarms))
                for s1 in range(self.num_swarms):
                    for s2 in range(s1 + 1, self.num_swarms):
                        swarm_distances[s1, s2] = np.linalg.norm(self.swarm_best_positions[s1, np.argmin(self.swarm_best_fitness[s1])] - self.swarm_best_positions[s2, np.argmin(self.swarm_best_fitness[s2])])
                        swarm_distances[s2, s1] = swarm_distances[s1, s2]
                
                s1, s2 = np.unravel_index(np.argmin(swarm_distances, axis=None), swarm_distances.shape)
                
                # Merge swarm s2 into s1: append s2 particles to s1 and remove s2
                self.swarms[s1] = np.concatenate((self.swarms[s1], self.swarms[s2]), axis=0)
                self.swarm_fitness[s1] = np.concatenate((self.swarm_fitness[s1], self.swarm_fitness[s2]), axis=0)
                self.swarm_best_positions[s1] = np.concatenate((self.swarm_best_positions[s1], self.swarm_best_positions[s2]), axis=0)
                self.swarm_best_fitness[s1] = np.concatenate((self.swarm_best_fitness[s1], self.swarm_best_fitness[s2]), axis=0)
                
                # Re-evaluate merged swarm
                for k in range(self.swarms[s1].shape[0]):
                    fitness = func(self.swarms[s1][k])
                    self.eval_count += 1
                    self.swarm_fitness[s1][k] = fitness
                    if fitness < self.swarm_best_fitness[s1][k]:
                        self.swarm_best_fitness[s1][k] = fitness
                        self.swarm_best_positions[s1][k] = np.copy(self.swarms[s1][k])
                    if fitness < self.global_best_fitness:
                        self.global_best_fitness = fitness
                        self.global_best_position = np.copy(self.swarms[s1][k])
                    if self.eval_count >= self.budget:
                        break
                if self.eval_count >= self.budget:
                    break

                # Reduce number of swarms
                self.num_swarms -=1
                # Remove swarm s2 from other swarm arrays
                indices = [x for x in range(self.swarms.shape[0]) if x != s2]
                self.swarms = self.swarms[indices]
                self.swarm_fitness = self.swarm_fitness[indices]
                self.swarm_best_positions = self.swarm_best_positions[indices]
                self.swarm_best_fitness = self.swarm_best_fitness[indices]

        return self.global_best_fitness, self.global_best_position