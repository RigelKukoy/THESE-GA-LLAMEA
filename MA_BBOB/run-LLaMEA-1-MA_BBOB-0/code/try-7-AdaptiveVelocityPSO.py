import numpy as np

class AdaptiveVelocityPSO:
    def __init__(self, budget=10000, dim=10, pop_size_initial=20, c1=1.49, c2=1.49, w_max=0.9, w_min=0.4, local_search_iterations=5, stagnation_threshold=500):
        self.budget = budget
        self.dim = dim
        self.pop_size_initial = pop_size_initial
        self.pop_size = pop_size_initial  # Current population size
        self.c1 = c1
        self.c2 = c2
        self.w_max = w_max
        self.w_min = w_min
        self.local_search_iterations = local_search_iterations
        self.stagnation_threshold = stagnation_threshold
        self.stagnation_counter = 0
        self.restart_percentage = 0.5
        self.expansion_rate = 1.2  # Rate at which population size increases after stagnation
        self.contraction_rate = 0.8  # Rate at which population size decreases if improving
        self.min_pop_size = 10
        self.max_pop_size = 50
        self.last_improvement = 0 # Keep track of the last time gbest improved
        self.f_opt = np.Inf
        self.x_opt = None


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
        self.last_improvement = 0 # Initialize last improvement

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
                        self.stagnation_counter = 0  # Reset stagnation counter upon improvement
                        self.last_improvement = eval_count

                #Local Search with Adaptive Step Size
                if eval_count < self.budget:
                    x_current = self.population[i].copy()
                    f_current = fitness
                    adaptive_step_size = 0.01 * (func.bounds.ub - func.bounds.lb)

                    for _ in range(self.local_search_iterations):
                        perturbation = np.random.normal(0, adaptive_step_size, size=self.dim)
                        x_new = x_current + perturbation
                        x_new = np.clip(x_new, func.bounds.lb, func.bounds.ub)
                        f_new = func(x_new)
                        eval_count += 1

                        if f_new < f_current:
                            x_current = x_new.copy()
                            f_current = f_new
                            adaptive_step_size *= 1.1 # Increase step if improvement
                        else:
                            adaptive_step_size *= 0.9 # Reduce step if no improvement

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
                            self.stagnation_counter = 0
                            self.last_improvement = eval_count

            #Stagnation Restart Mechanism and Dynamic Population Size
            if eval_count - self.last_improvement > self.stagnation_threshold:
                #Stagnation is detected
                if self.pop_size < self.max_pop_size:
                    self.pop_size = min(int(self.pop_size * self.expansion_rate), self.max_pop_size) #Increase population
                
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
                        self.last_improvement = eval_count
                        
                self.stagnation_counter = 0 # Reset stagnation counter after restart
                self.last_improvement = eval_count

            else:
                if self.gbest_fitness < self.f_opt:
                    self.f_opt = self.gbest_fitness
                    self.x_opt = self.gbest_position.copy()
                    # Potentially decrease pop size if improvement is good to save budget
                    if self.pop_size > self.min_pop_size:
                        self.pop_size = max(int(self.pop_size * self.contraction_rate), self.min_pop_size)
            

            if self.gbest_fitness < self.f_opt:
                self.f_opt = self.gbest_fitness
                self.x_opt = self.gbest_position.copy()

            if eval_count >= self.budget:
                break

        return self.f_opt, self.x_opt