import numpy as np

class AdaptiveVelocityPSO:
    def __init__(self, budget=10000, dim=10, pop_size=20, c1_init=2.0, c2_init=2.0, w_init=0.9, w_final=0.4, local_search_iterations=5, stagnation_threshold=500, restart_percentage=0.3, orthogonal_learning_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.c1_init = c1_init
        self.c2_init = c2_init
        self.w_init = w_init
        self.w_final = w_final
        self.local_search_iterations = local_search_iterations
        self.stagnation_threshold = stagnation_threshold
        self.restart_percentage = restart_percentage
        self.orthogonal_learning_rate = orthogonal_learning_rate
        self.eval_count = 0 # Track evaluations
        self.v_max_factor = 0.1 # Maximum velocity factor
        

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        lb, ub = func.bounds.lb, func.bounds.ub
        self.population = np.random.uniform(lb, ub, size=(self.pop_size, self.dim))
        self.velocities = np.random.uniform(-self.v_max_factor * (ub - lb), self.v_max_factor * (ub - lb), size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.eval_count += self.pop_size

        self.pbest_positions = self.population.copy()
        self.pbest_fitness = self.fitness.copy()
        self.gbest_index = np.argmin(self.fitness)
        self.gbest_position = self.population[self.gbest_index].copy()
        self.gbest_fitness = self.fitness[self.gbest_index]
        self.stagnation_counter = 0

        while self.eval_count < self.budget:
            # Dynamic Parameter Adaptation
            w = self.w_init - (self.w_init - self.w_final) * (self.eval_count / self.budget)
            c1 = self.c1_init - (self.c1_init - 0.5) * (self.eval_count / self.budget) #Decay c1
            c2 = self.c2_init + (2.5 - self.c2_init) * (self.eval_count / self.budget)  #Increase c2

            for i in range(self.pop_size):
                # Update Velocity
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                self.velocities[i] = w * self.velocities[i] + \
                                     c1 * r1 * (self.pbest_positions[i] - self.population[i]) + \
                                     c2 * r2 * (self.gbest_position - self.population[i])

                # Velocity Clamping
                v_max = self.v_max_factor * (ub - lb)
                self.velocities[i] = np.clip(self.velocities[i], -v_max, v_max)

                # Update Position
                self.population[i] = self.population[i] + self.velocities[i]
                self.population[i] = np.clip(self.population[i], lb, ub) # Boundary Handling

                # Evaluate Fitness
                fitness = func(self.population[i])
                self.eval_count += 1

                # Update Personal Best
                if fitness < self.pbest_fitness[i]:
                    self.pbest_fitness[i] = fitness
                    self.pbest_positions[i] = self.population[i].copy()

                    # Update Global Best
                    if fitness < self.gbest_fitness:
                        self.gbest_fitness = fitness
                        self.gbest_position = self.population[i].copy()
                        self.stagnation_counter = 0  # Reset stagnation counter
                else:
                    self.stagnation_counter += 1
                
                # Local Search
                if self.eval_count < self.budget:
                    x_current = self.population[i].copy()
                    f_current = fitness
                    for _ in range(self.local_search_iterations):
                        perturbation = np.random.normal(0, 0.01 * (ub - lb), size=self.dim)
                        x_new = x_current + perturbation
                        x_new = np.clip(x_new, lb, ub)
                        f_new = func(x_new)
                        self.eval_count += 1
                        if f_new < f_current:
                            x_current = x_new.copy()
                            f_current = f_new
                        if self.eval_count >= self.budget:
                            break
                            
                    self.population[i] = x_current
                    self.fitness[i] = f_current

                    if f_current < self.pbest_fitness[i]:
                        self.pbest_fitness[i] = f_current
                        self.pbest_positions[i] = self.population[i].copy()

                        if f_current < self.gbest_fitness:
                            self.gbest_fitness = f_current
                            self.gbest_position = self.population[i].copy()
                            self.stagnation_counter = 0

                # Orthogonal Learning
                if self.eval_count < self.budget:
                    orthogonal_vector = np.random.normal(0, 0.005 * (ub-lb), size=self.dim)
                    x_orthogonal = self.population[i] + self.orthogonal_learning_rate * orthogonal_vector
                    x_orthogonal = np.clip(x_orthogonal, lb, ub)
                    f_orthogonal = func(x_orthogonal)
                    self.eval_count +=1

                    if f_orthogonal < self.fitness[i]:
                        self.population[i] = x_orthogonal
                        self.fitness[i] = f_orthogonal
                        if f_orthogonal < self.pbest_fitness[i]:
                            self.pbest_fitness[i] = f_orthogonal
                            self.pbest_positions[i] = self.population[i].copy()
                            if f_orthogonal < self.gbest_fitness:
                                self.gbest_fitness = f_orthogonal
                                self.gbest_position = self.population[i].copy()
                                self.stagnation_counter = 0


            # Stagnation Restart
            if self.stagnation_counter > self.stagnation_threshold:
                num_to_restart = int(self.restart_percentage * self.pop_size)
                indices_to_restart = np.random.choice(self.pop_size, size=num_to_restart, replace=False)

                for idx in indices_to_restart:
                    self.population[idx] = np.random.uniform(lb, ub, size=self.dim)
                    self.velocities[idx] = np.random.uniform(-self.v_max_factor * (ub - lb), self.v_max_factor * (ub - lb), size=self.dim)
                    self.fitness[idx] = func(self.population[idx])
                    self.eval_count += 1

                    self.pbest_positions[idx] = self.population[idx].copy()
                    self.pbest_fitness[idx] = self.fitness[idx]

                    if self.fitness[idx] < self.gbest_fitness:
                        self.gbest_fitness = self.fitness[idx]
                        self.gbest_position = self.population[idx].copy()

                self.stagnation_counter = 0

            if self.gbest_fitness < self.f_opt:
                self.f_opt = self.gbest_fitness
                self.x_opt = self.gbest_position.copy()
            
            if self.eval_count >= self.budget:
                break

        return self.f_opt, self.x_opt