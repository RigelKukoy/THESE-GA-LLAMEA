import numpy as np

class AdaptiveVelocityPSO_Ortho(object):
    def __init__(self, budget=10000, dim=10, pop_size=20, c1=1.49, c2=1.49, w_max=0.9, w_min=0.4, local_search_iterations=5, ortho_group_size=5):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.c1 = c1
        self.c2 = c2
        self.w_max = w_max
        self.w_min = w_min
        self.local_search_iterations = local_search_iterations
        self.stagnation_threshold = 1000
        self.restart_percentage = 0.5
        self.ortho_group_size = min(ortho_group_size, pop_size) # Ensure ortho_group_size is valid

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, size=(self.pop_size, self.dim)) * 0.1 * (func.bounds.ub - func.bounds.lb)
        self.fitness = np.array([func(x) for x in self.population])
        self.pbest_positions = self.population.copy()
        self.pbest_fitness = self.fitness.copy()
        self.gbest_index = np.argmin(self.fitness)
        self.gbest_position = self.population[self.gbest_index].copy()
        self.gbest_fitness = self.fitness[self.gbest_index]

        eval_count = self.pop_size
        stagnation_counter = 0
        adaptive_c1 = self.c1
        adaptive_c2 = self.c2

        while eval_count < self.budget:
            # Adaptive Inertia Weight
            w = self.w_max - (self.w_max - self.w_min) * (eval_count / self.budget)

            for i in range(self.pop_size):
                # Update Velocity
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                self.velocities[i] = w * self.velocities[i] + \
                                     adaptive_c1 * r1 * (self.pbest_positions[i] - self.population[i]) + \
                                     adaptive_c2 * r2 * (self.gbest_position - self.population[i])

                # Limit velocity to avoid explosion
                v_max = 0.1 * (func.bounds.ub - func.bounds.lb)
                self.velocities[i] = np.clip(self.velocities[i], -v_max, v_max)

                # Update Position
                self.population[i] = self.population[i] + self.velocities[i]

                # Boundary Handling (Clipping)
                self.population[i] = np.clip(self.population[i], func.bounds.lb, func.bounds.ub)

                # Evaluate Fitness
                fitness = func(self.population[i])
                eval_count += 1

                # Update Personal Best
                if fitness < self.pbest_fitness[i]:
                    self.pbest_fitness[i] = fitness
                    self.pbest_positions[i] = self.population[i].copy()

                    # Update Global Best
                    if fitness < self.gbest_fitness:
                        self.gbest_fitness = fitness
                        self.gbest_position = self.population[i].copy()
                        stagnation_counter = 0  # Reset stagnation counter upon improvement
                        adaptive_c1 = self.c1 # Reset cognitive parameter on global best improvement
                        adaptive_c2 = self.c2 # Reset social parameter on global best improvement

                else:
                    stagnation_counter += 1  # Increment stagnation counter
                    adaptive_c1 *= 0.99  # Reduce cognitive parameter if no improvement
                    adaptive_c2 *= 1.01  # Increase social parameter if no improvement
                    adaptive_c1 = np.clip(adaptive_c1, 0.5, 2.0) # Keep parameters in reasonable range
                    adaptive_c2 = np.clip(adaptive_c2, 0.5, 2.0) # Keep parameters in reasonable range
                
                #Local Search
                if eval_count < self.budget:
                     x_current = self.population[i].copy()
                     f_current = fitness

                     for _ in range(self.local_search_iterations):
                         perturbation = np.random.normal(0, 0.01 * (func.bounds.ub - func.bounds.lb), size=self.dim)
                         x_new = x_current + perturbation
                         x_new = np.clip(x_new, func.bounds.lb, func.bounds.ub)
                         f_new = func(x_new)
                         eval_count+=1

                         if f_new < f_current:
                            x_current = x_new.copy()
                            f_current = f_new

                         if eval_count >= self.budget:
                            break
                     
                     self.population[i] = x_current
                     self.fitness[i] = f_current
                
                     if f_current < self.pbest_fitness[i]:
                         self.pbest_fitness[i] = f_current
                         self.pbest_positions[i] = self.population[i].copy()

                         if f_current < self.gbest_fitness:
                            self.gbest_fitness = f_current
                            self.gbest_position = self.population[i].copy()
                            stagnation_counter = 0

            # Orthogonal Learning
            if self.pop_size > 1 and eval_count < self.budget:
                indices = np.random.choice(self.pop_size, size=self.ortho_group_size, replace=False)
                group = self.population[indices]
                
                # Calculate the mean position of the group
                mean_position = np.mean(group, axis=0)

                # Generate an orthogonal vector (simplified - could be improved for higher dimensions)
                orthogonal_vector = np.random.normal(0, 0.05 * (func.bounds.ub - func.bounds.lb), size=self.dim)
                
                # Update a randomly selected particle in the group with the orthogonal information
                idx_to_update = np.random.randint(0, self.ortho_group_size)
                new_position = mean_position + orthogonal_vector
                new_position = np.clip(new_position, func.bounds.lb, func.bounds.ub)
                new_fitness = func(new_position)
                eval_count += 1

                if new_fitness < self.fitness[indices[idx_to_update]]:
                    self.population[indices[idx_to_update]] = new_position
                    self.fitness[indices[idx_to_update]] = new_fitness

                    if new_fitness < self.pbest_fitness[indices[idx_to_update]]:
                        self.pbest_fitness[indices[idx_to_update]] = new_fitness
                        self.pbest_positions[indices[idx_to_update]] = new_position.copy()

                        if new_fitness < self.gbest_fitness:
                            self.gbest_fitness = new_fitness
                            self.gbest_position = new_position.copy()
                            stagnation_counter = 0
            

            # Stagnation Restart Mechanism
            if stagnation_counter > self.stagnation_threshold:
                num_to_restart = int(self.restart_percentage * self.pop_size)
                indices_to_restart = np.random.choice(self.pop_size, size=num_to_restart, replace=False)

                for idx in indices_to_restart:
                    self.population[idx] = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
                    self.velocities[idx] = np.random.uniform(-1, 1, size=self.dim) * 0.1 * (func.bounds.ub - func.bounds.lb)
                    self.fitness[idx] = func(self.population[idx])
                    eval_count += 1

                    self.pbest_positions[idx] = self.population[idx].copy()
                    self.pbest_fitness[idx] = self.fitness[idx]

                    if self.fitness[idx] < self.gbest_fitness:
                        self.gbest_fitness = self.fitness[idx]
                        self.gbest_position = self.population[idx].copy()

                stagnation_counter = 0  # Reset stagnation counter after restart

            if self.gbest_fitness < self.f_opt:
                self.f_opt = self.gbest_fitness
                self.x_opt = self.gbest_position.copy()

            if eval_count >= self.budget:
                break

        return self.f_opt, self.x_opt