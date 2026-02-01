import numpy as np

class AdaptiveDECauchyRestart:
    def __init__(self, budget=10000, dim=10, pop_size=50, CR_init=0.5, F_init=0.7, cauchy_scale=0.1, diversity_threshold=0.01):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.CR_init = CR_init
        self.F_init = F_init
        self.cauchy_scale = cauchy_scale
        self.diversity_threshold = diversity_threshold
        self.population = None
        self.fitness = None
        self.CR = None
        self.F = None
        self.f_opt = np.Inf
        self.x_opt = None
        self.eval_count = 0
        self.stagnation_counter = 0

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.eval_count += self.pop_size
        self.CR = np.full(self.pop_size, self.CR_init)
        self.F = np.full(self.pop_size, self.F_init)
        
        best_index = np.argmin(self.fitness)
        if self.fitness[best_index] < self.f_opt:
            self.f_opt = self.fitness[best_index]
            self.x_opt = self.population[best_index].copy()

    def cauchy_mutation(self, x_r1, x_r2, x_r3):
        delta = x_r2 - x_r3
        cauchy_noise = np.random.standard_cauchy(size=self.dim) * self.cauchy_scale
        return x_r1 + self.F * delta + cauchy_noise

    def evolve(self, func):
        f_opt_old = self.f_opt
        for i in range(self.pop_size):
            # Mutation
            idxs = np.random.choice(self.pop_size, 3, replace=False)
            x_r1, x_r2, x_r3 = self.population[idxs]
            x_mutated = self.cauchy_mutation(x_r1, x_r2, x_r3) # Use Cauchy mutation
            x_mutated = np.clip(x_mutated, func.bounds.lb, func.bounds.ub)

            # Crossover
            x_trial = self.population[i].copy()
            j_rand = np.random.randint(self.dim)
            CR = np.clip(np.random.normal(self.CR[i], 0.1), 0.0, 1.0)  # Self-adjusting CR
            F = np.clip(np.random.normal(self.F[i], 0.1), 0.1, 2.0) # Self-adjusting F
            
            for j in range(self.dim):
                if np.random.rand() < CR or j == j_rand:
                    x_trial[j] = x_mutated[j]

            # Selection
            f_trial = func(x_trial)
            self.eval_count += 1

            if f_trial < self.fitness[i]:
                self.CR[i] = CR  # Update CR of individual
                self.F[i] = F # Update F of individual
                self.population[i] = x_trial
                self.fitness[i] = f_trial
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = x_trial.copy()
            
            if self.eval_count >= self.budget:
                break
        
        # Restart mechanism based on population diversity
        diversity = np.std(self.fitness)
        if diversity < self.diversity_threshold:
            self.stagnation_counter +=1
        else:
            self.stagnation_counter = 0
        
        if self.stagnation_counter > 20:
             self.initialize_population(func) # restart population
             self.stagnation_counter = 0

    def __call__(self, func):
        self.initialize_population(func)
        while self.eval_count < self.budget:
            self.evolve(func)
        return self.f_opt, self.x_opt