import numpy as np

class AdaptiveVelocityPSO_Enhanced:
    def __init__(self, budget=10000, dim=10, pop_size=20, c1=1.5, c2=1.5, w_max=0.9, w_min=0.4, local_search_iterations=5, stagnation_threshold=500, stagnation_multiplier=1.2):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.c1 = c1
        self.c2 = c2
        self.w_max = w_max
        self.w_min = w_min
        self.local_search_iterations = local_search_iterations
        self.stagnation_threshold = stagnation_threshold
        self.stagnation_multiplier = stagnation_multiplier #Factor to increase threshold
        self.restart_percentage = 0.7
        self.exploration_probability = 0.1
        self.orthogonal_learning_percentage = 0.2
        self.min_velocity = -0.1 * (5.0 - (-5.0))
        self.max_velocity = 0.1 * (5.0 - (-5.0))
        self.orthogonal_learning_rate = 0.05 # Initial learning rate for orthogonal learning
        self.orthogonal_learning_decay = 0.95 # Decay rate for orthogonal learning


    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        lb = func.bounds.lb
        ub = func.bounds.ub
        self.population = np.random.uniform(lb, ub, size=(self.pop_size, self.dim))
        self.velocities = np.random.uniform(self.min_velocity, self.max_velocity, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.pbest_positions = self.population.copy()
        self.pbest_fitness = self.fitness.copy()
        self.gbest_index = np.argmin(self.fitness)
        self.gbest_position = self.population[self.gbest_index].copy()
        self.gbest_fitness = self.fitness[self.gbest_index]

        eval_count = self.pop_size
        stagnation_counter = 0
        current_stagnation_threshold = self.stagnation_threshold

        while eval_count < self.budget:
            # Adaptive Inertia Weight
            w = self.w_max - (self.w_max - self.w_min) * (eval_count / self.budget)

            # Adaptive Acceleration Coefficients
            c1 = self.c1 + (1 - (eval_count / self.budget)) * 0.5
            c2 = self.c2 + (eval_count / self.budget) * 0.5

            for i in range(self.pop_size):
                # Exploration Move
                if np.random.rand() < self.exploration_probability:
                    self.population[i] = np.random.uniform(lb, ub, size=self.dim)
                    self.velocities[i] = np.random.uniform(self.min_velocity, self.max_velocity, size=self.dim)
                else:
                    # Update Velocity
                    r1 = np.random.rand(self.dim)
                    r2 = np.random.rand(self.dim)
                    self.velocities[i] = w * self.velocities[i] + \
                                         c1 * r1 * (self.pbest_positions[i] - self.population[i]) + \
                                         c2 * r2 * (self.gbest_position - self.population[i])

                    # Limit velocity
                    self.velocities[i] = np.clip(self.velocities[i], self.min_velocity, self.max_velocity)

                    # Update Position
                    self.population[i] = self.population[i] + self.velocities[i]

                    # Boundary Handling
                    self.population[i] = np.clip(self.population[i], lb, ub)


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
                        stagnation_counter = 0

                else:
                    stagnation_counter += 1

                # Local Search
                if eval_count < self.budget:
                    x_current = self.population[i].copy()
                    f_current = fitness

                    for _ in range(self.local_search_iterations):
                        perturbation = np.random.normal(0, 0.01 * (ub - lb), size=self.dim)
                        x_new = x_current + perturbation
                        x_new = np.clip(x_new, lb, ub)
                        f_new = func(x_new)
                        eval_count += 1

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
            num_orthogonal = int(self.orthogonal_learning_percentage * self.pop_size)
            indices_orthogonal = np.random.choice(self.pop_size, size=num_orthogonal, replace=False)

            for idx in indices_orthogonal:
                basis = np.random.normal(0, 1, size=(self.dim, self.dim))
                Q, _ = np.linalg.qr(basis)  # Orthogonal basis

                step_sizes = np.random.uniform(-self.orthogonal_learning_rate * (ub - lb), self.orthogonal_learning_rate * (ub - lb), size=self.dim)  # Smaller step sizes, decaying

                for j in range(self.dim):
                    x_new = self.population[idx] + step_sizes[j] * Q[:, j]
                    x_new = np.clip(x_new, lb, ub)
                    f_new = func(x_new)
                    eval_count += 1

                    if f_new < self.fitness[idx]:
                        self.population[idx] = x_new.copy()
                        self.fitness[idx] = f_new

                        if f_new < self.pbest_fitness[idx]:
                            self.pbest_fitness[idx] = f_new
                            self.pbest_positions[idx] = self.population[idx].copy()

                            if f_new < self.gbest_fitness:
                                self.gbest_fitness = f_new
                                self.gbest_position = self.population[idx].copy()
                                stagnation_counter = 0

                    if eval_count >= self.budget:
                        break

                if eval_count >= self.budget:
                    break
                
            self.orthogonal_learning_rate *= self.orthogonal_learning_decay # Decay learning rate

            # Stagnation Restart Mechanism
            if stagnation_counter > current_stagnation_threshold:
                num_to_restart = int(self.restart_percentage * self.pop_size)
                indices_to_restart = np.random.choice(self.pop_size, size=num_to_restart, replace=False)

                for idx in indices_to_restart:
                    self.population[idx] = np.random.uniform(lb, ub, size=self.dim)
                    self.velocities[idx] = np.random.uniform(self.min_velocity, self.max_velocity, size=self.dim)
                    self.fitness[idx] = func(self.population[idx])
                    eval_count += 1

                    self.pbest_positions[idx] = self.population[idx].copy()
                    self.pbest_fitness[idx] = self.fitness[idx]

                    if self.fitness[idx] < self.gbest_fitness:
                        self.gbest_fitness = self.fitness[idx]
                        self.gbest_position = self.population[idx].copy()

                stagnation_counter = 0
                current_stagnation_threshold *= self.stagnation_multiplier  # Increase stagnation threshold

            if self.gbest_fitness < self.f_opt:
                self.f_opt = self.gbest_fitness
                self.x_opt = self.gbest_position.copy()

            if eval_count >= self.budget:
                break

        return self.f_opt, self.x_opt