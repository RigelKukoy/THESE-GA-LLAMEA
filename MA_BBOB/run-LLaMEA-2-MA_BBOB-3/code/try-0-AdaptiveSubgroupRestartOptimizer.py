import numpy as np

class AdaptiveSubgroupRestartOptimizer:
    def __init__(self, budget=10000, dim=10, pop_size=30, num_subgroups=3, stagnation_tolerance=1000, pso_inertia=0.7, pso_cognitive=1.4, pso_social=1.4, de_mutation=0.5, local_search_prob=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.num_subgroups = num_subgroups
        self.stagnation_tolerance = stagnation_tolerance
        self.pso_inertia = pso_inertia
        self.pso_cognitive = pso_cognitive
        self.pso_social = pso_social
        self.de_mutation = de_mutation
        self.local_search_prob = local_search_prob
        self.population = np.random.uniform(-5, 5, size=(pop_size, dim))
        self.fitness = np.zeros(pop_size)
        self.velocities = np.zeros_like(self.population)
        self.best_positions = self.population.copy()
        self.best_fitness = np.full(pop_size, np.inf)
        self.global_best_position = None
        self.global_best_fitness = np.inf
        self.eval_count = 0
        self.stagnation_counter = 0
        self.last_best_fitness = np.inf

    def __call__(self, func):
        self.eval_count = 0
        self.stagnation_counter = 0
        self.last_best_fitness = np.inf
        
        while self.eval_count < self.budget:
            # Evaluate fitness
            for i in range(self.pop_size):
                if self.eval_count < self.budget:
                    f = func(self.population[i])
                    self.eval_count += 1
                    self.fitness[i] = f
                    if f < self.best_fitness[i]:
                        self.best_fitness[i] = f
                        self.best_positions[i] = self.population[i].copy()
                        if f < self.global_best_fitness:
                            self.global_best_fitness = f
                            self.global_best_position = self.population[i].copy()

            # Check for stagnation
            if self.global_best_fitness >= self.last_best_fitness:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0

            self.last_best_fitness = self.global_best_fitness

            # Restart if stagnated
            if self.stagnation_counter > self.stagnation_tolerance:
                self.population = np.random.uniform(-5, 5, size=(self.pop_size, self.dim))
                self.velocities = np.zeros_like(self.population)
                self.fitness = np.zeros(self.pop_size)
                self.best_positions = self.population.copy()
                self.best_fitness = np.full(self.pop_size, np.inf)
                self.stagnation_counter = 0
                continue

            # Subgrouping and Update
            subgroup_size = self.pop_size // self.num_subgroups
            for subgroup_id in range(self.num_subgroups):
                start_index = subgroup_id * subgroup_size
                end_index = (subgroup_id + 1) * subgroup_size if subgroup_id < self.num_subgroups - 1 else self.pop_size

                # PSO update for this subgroup
                for i in range(start_index, end_index):
                    r1 = np.random.rand(self.dim)
                    r2 = np.random.rand(self.dim)
                    self.velocities[i] = (self.pso_inertia * self.velocities[i] +
                                          self.pso_cognitive * r1 * (self.best_positions[i] - self.population[i]) +
                                          self.pso_social * r2 * (self.global_best_position - self.population[i]))
                    self.population[i] = np.clip(self.population[i] + self.velocities[i], func.bounds.lb, func.bounds.ub)
                    
                    # DE update with random individuals from the population
                    r1, r2, r3 = np.random.choice(self.pop_size, 3, replace=False)
                    de_vector = self.population[i] + self.de_mutation * (self.population[r1] - self.population[r2])
                    de_vector = np.clip(de_vector, func.bounds.lb, func.bounds.ub)
                    
                    # Local search
                    if np.random.rand() < self.local_search_prob:
                        perturbation = np.random.uniform(-0.1, 0.1, size=self.dim)
                        local_search_vector = self.population[i] + perturbation
                        local_search_vector = np.clip(local_search_vector, func.bounds.lb, func.bounds.ub)
                    
                        f_de = func(de_vector) if self.eval_count + 1 < self.budget else np.inf
                        f_ls = func(local_search_vector) if self.eval_count + 1 < self.budget else np.inf
                    
                        if self.eval_count + 2 < self.budget:
                            self.eval_count += 2
                            if f_de < f_ls and f_de < self.fitness[i]:
                                self.population[i] = de_vector
                                self.fitness[i] = f_de
                                if f_de < self.best_fitness[i]:
                                    self.best_fitness[i] = f_de
                                    self.best_positions[i] = self.population[i].copy()
                                    if f_de < self.global_best_fitness:
                                        self.global_best_fitness = f_de
                                        self.global_best_position = self.population[i].copy()
                            elif f_ls < self.fitness[i]:
                                self.population[i] = local_search_vector
                                self.fitness[i] = f_ls
                                if f_ls < self.best_fitness[i]:
                                    self.best_fitness[i] = f_ls
                                    self.best_positions[i] = self.population[i].copy()
                                    if f_ls < self.global_best_fitness:
                                        self.global_best_fitness = f_ls
                                        self.global_best_position = self.population[i].copy()
                        else:
                            break # break if almost budget exhausted
                    else:
                        f_de = func(de_vector) if self.eval_count + 1 < self.budget else np.inf
                        if self.eval_count + 1 < self.budget:
                            self.eval_count += 1
                            if f_de < self.fitness[i]:
                                self.population[i] = de_vector
                                self.fitness[i] = f_de
                                if f_de < self.best_fitness[i]:
                                    self.best_fitness[i] = f_de
                                    self.best_positions[i] = self.population[i].copy()
                                    if f_de < self.global_best_fitness:
                                        self.global_best_fitness = f_de
                                        self.global_best_position = self.population[i].copy()
                        else:
                            break
                        
        return self.global_best_fitness, self.global_best_position