import numpy as np

class AdaptiveVelocityPSO:
    def __init__(self, budget=10000, dim=10, pop_size=20, c1=1.49, c2=1.49, w_max=0.9, w_min=0.4, local_search_iterations=5, stagnation_threshold=500, restart_percentage=0.5, orthogonal_learning_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.c1 = c1
        self.c2 = c2
        self.w_max = w_max
        self.w_min = w_min
        self.local_search_iterations = local_search_iterations
        self.stagnation_threshold = stagnation_threshold # Number of iterations without improvement before restart
        self.restart_percentage = restart_percentage # Percentage of population to restart
        self.orthogonal_learning_rate = orthogonal_learning_rate # Learning rate for orthogonal learning
        self.success_history = [] # Keep track of success for parameter adaptation
        self.success_threshold = 0.1 # Threshold for considering an update successful

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, size=(self.pop_size, self.dim)) * 0.1 * (func.bounds.ub - func.bounds.lb)  # Initialize velocities
        self.fitness = np.array([func(x) for x in self.population])
        self.pbest_positions = self.population.copy()
        self.pbest_fitness = self.fitness.copy()
        self.gbest_index = np.argmin(self.fitness)
        self.gbest_position = self.population[self.gbest_index].copy()
        self.gbest_fitness = self.fitness[self.gbest_index]

        eval_count = self.pop_size
        stagnation_counter = 0
        last_improvement = 0 # Keep track of the last time the global best improved

        while eval_count < self.budget:
            # Adaptive Inertia Weight
            w = self.w_max - (self.w_max - self.w_min) * (eval_count / self.budget)

            for i in range(self.pop_size):
                # Update Velocity
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                self.velocities[i] = w * self.velocities[i] + \
                                     self.c1 * r1 * (self.pbest_positions[i] - self.population[i]) + \
                                     self.c2 * r2 * (self.gbest_position - self.population[i])

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

                    #Update Global Best
                    if fitness < self.gbest_fitness:
                        self.gbest_fitness = fitness
                        self.gbest_position = self.population[i].copy()
                        last_improvement = eval_count # Update last improvement time

                
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
                            last_improvement = eval_count

            # Orthogonal Learning
            if eval_count < self.budget:
                orthogonal_vector = np.random.normal(0, 1, size=self.dim)
                orthogonal_vector /= np.linalg.norm(orthogonal_vector)  # Normalize

                # Move each particle along the orthogonal direction
                for i in range(self.pop_size):
                    step_size = self.orthogonal_learning_rate * (func.bounds.ub - func.bounds.lb)
                    x_new = self.population[i] + step_size * orthogonal_vector
                    x_new = np.clip(x_new, func.bounds.lb, func.bounds.ub)  # Boundary handling

                    f_new = func(x_new)
                    eval_count += 1

                    if f_new < self.fitness[i]:
                        self.population[i] = x_new.copy()
                        self.fitness[i] = f_new
                        if f_new < self.pbest_fitness[i]:
                           self.pbest_fitness[i] = f_new
                           self.pbest_positions[i] = self.population[i].copy()

                           if f_new < self.gbest_fitness:
                               self.gbest_fitness = f_new
                               self.gbest_position = self.population[i].copy()
                               last_improvement = eval_count

            # Stagnation Detection and Restart Mechanism (Improved)
            if eval_count - last_improvement > self.stagnation_threshold:
                # Restart strategy with enhanced exploration
                num_to_restart = int(self.restart_percentage * self.pop_size)
                indices_to_restart = np.random.choice(self.pop_size, size=num_to_restart, replace=False)

                for idx in indices_to_restart:
                    # Restart with a wider range and towards gbest
                    direction_to_gbest = self.gbest_position - self.population[idx]
                    self.population[idx] = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim) + 0.2 * direction_to_gbest
                    self.population[idx] = np.clip(self.population[idx], func.bounds.lb, func.bounds.ub)

                    self.velocities[idx] = np.random.uniform(-1, 1, size=self.dim) * 0.1 * (func.bounds.ub - func.bounds.lb)  # Reinitialize velocity
                    self.fitness[idx] = func(self.population[idx])
                    eval_count += 1

                    self.pbest_positions[idx] = self.population[idx].copy()
                    self.pbest_fitness[idx] = self.fitness[idx]

                    if self.fitness[idx] < self.gbest_fitness:
                        self.gbest_fitness = self.fitness[idx]
                        self.gbest_position = self.population[idx].copy()
                        last_improvement = eval_count

                last_improvement = eval_count  # Reset last improvement after restart

            # Parameter Adaptation (Example: Adjusting c1 and c2)
            if len(self.success_history) > 10:  # Only adapt if enough history exists
                success_rate = np.mean(self.success_history[-10:])
                if success_rate < self.success_threshold:
                    # Exploration is needed: increase c1 and decrease c2
                    self.c1 *= 1.05
                    self.c2 *= 0.95
                else:
                    # Exploitation is needed: decrease c1 and increase c2
                    self.c1 *= 0.95
                    self.c2 *= 1.05

                # Ensure parameters stay within reasonable bounds
                self.c1 = np.clip(self.c1, 1.0, 2.0)
                self.c2 = np.clip(self.c2, 1.0, 2.0)


            if self.gbest_fitness < self.f_opt:
                self.f_opt = self.gbest_fitness
                self.x_opt = self.gbest_position.copy()
                self.success_history.append(1) #Mark success
            else:
                self.success_history.append(0)  # Mark failure

            if eval_count >= self.budget:
                break

        return self.f_opt, self.x_opt