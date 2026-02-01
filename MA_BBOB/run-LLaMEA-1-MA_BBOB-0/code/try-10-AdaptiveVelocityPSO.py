import numpy as np

class AdaptiveVelocityPSO:
    def __init__(self, budget=10000, dim=10, pop_size=20, c1=1.49, c2=1.49, w_max=0.9, w_min=0.4, local_search_iterations=5,
                 mutation_rate=0.05, stagnation_threshold=1000, restart_percentage=0.5, age_threshold=500):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.c1 = c1
        self.c2 = c2
        self.w_max = w_max
        self.w_min = w_min
        self.local_search_iterations = local_search_iterations
        self.mutation_rate = mutation_rate
        self.stagnation_threshold = stagnation_threshold
        self.restart_percentage = restart_percentage
        self.age_threshold = age_threshold
        self.ages = np.zeros(pop_size) # Initialize ages

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

                # Mutation
                if np.random.rand() < self.mutation_rate:
                    mutation = np.random.normal(0, 0.05 * (func.bounds.ub - func.bounds.lb), size=self.dim)
                    self.population[i] += mutation
                    self.population[i] = np.clip(self.population[i], func.bounds.lb, func.bounds.ub)

                # Evaluate Fitness
                fitness = func(self.population[i])
                eval_count += 1

                # Update Personal Best
                if fitness < self.pbest_fitness[i]:
                    self.pbest_fitness[i] = fitness
                    self.pbest_positions[i] = self.population[i].copy()
                    self.ages[i] = 0  # Reset age
                    #Update Global Best
                    if fitness < self.gbest_fitness:
                        self.gbest_fitness = fitness
                        self.gbest_position = self.population[i].copy()
                        stagnation_counter = 0  # Reset stagnation counter upon improvement

                else:
                    stagnation_counter += 1 # Increment stagnation counter
                    self.ages[i] += 1

                #Dynamic Local Search
                local_search_iterations = int(self.local_search_iterations * (1 + self.ages[i] / self.age_threshold))
                local_search_iterations = min(local_search_iterations, 20) # Cap the number of local search iterations
                if eval_count < self.budget:
                     x_current = self.population[i].copy()
                     f_current = fitness

                     for _ in range(local_search_iterations):
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
                         self.ages[i] = 0 # Reset age

                         if f_current < self.gbest_fitness:
                            self.gbest_fitness = f_current
                            self.gbest_position = self.population[i].copy()
                            stagnation_counter = 0

            #Stagnation Restart Mechanism
            if stagnation_counter > self.stagnation_threshold:
                num_to_restart = int(self.restart_percentage * self.pop_size)
                indices_to_restart = np.random.choice(self.pop_size, size=num_to_restart, replace=False)

                for idx in indices_to_restart:
                    self.population[idx] = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
                    self.velocities[idx] = np.random.uniform(-1, 1, size=self.dim) * 0.1 * (func.bounds.ub - func.bounds.lb) #Reinitialize velocity
                    self.fitness[idx] = func(self.population[idx])
                    self.ages[idx] = 0
                    eval_count += 1

                    self.pbest_positions[idx] = self.population[idx].copy()
                    self.pbest_fitness[idx] = self.fitness[idx]

                    if self.fitness[idx] < self.gbest_fitness:
                        self.gbest_fitness = self.fitness[idx]
                        self.gbest_position = self.population[idx].copy()
                        stagnation_counter = 0

                stagnation_counter = 0 # Reset stagnation counter after restart
            
            if self.gbest_fitness < self.f_opt:
                self.f_opt = self.gbest_fitness
                self.x_opt = self.gbest_position.copy()

            if eval_count >= self.budget:
                break

        return self.f_opt, self.x_opt