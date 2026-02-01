import numpy as np

class OrthogonalRestartDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, restart_trigger=0.01):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.restart_trigger = restart_trigger
        self.F = 0.5
        self.CR = 0.9
        self.best_fitness_history = []

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        
        for i in range(self.pop_size):
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = self.population[i]
        
        self.best_fitness_history.append(self.f_opt)

        while self.budget > 0:
            for i in range(self.pop_size):
                # Diversity-Guided Mutation
                if np.std(fitness) > self.restart_trigger:  # High diversity, encourage exploration
                    idxs = np.random.choice(self.pop_size, 3, replace=False)
                    x_r1, x_r2, x_r3 = self.population[idxs]
                    mutant = x_r1 + self.F * (x_r2 - x_r3)  # Classical DE mutation
                else:  # Low diversity, orthogonal learning
                    num_samples = min(self.dim, self.budget)  # Sample size for orthogonal design
                    if num_samples <= 0:
                       break
                    orthogonal_basis = self.generate_orthogonal_array(num_samples)
                    mutant = self.population[i].copy()
                    for j in range(num_samples):
                        pertubation = orthogonal_basis[j] * np.random.uniform(-self.F, self.F)
                        mutant[j % self.dim] += pertubation
                    mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)


                # Crossover
                crossover = np.random.uniform(size=self.dim) < self.CR
                trial = np.where(crossover, mutant, self.population[i])

                # Selection
                f_trial = func(trial)
                self.budget -= 1
                
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial

                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    self.population[i] = trial

            # Restart Mechanism
            if len(self.best_fitness_history) > 10 and np.std(self.best_fitness_history[-10:]) < self.restart_trigger:
                # Stagnation detected, restart a portion of the population
                num_to_restart = int(self.pop_size * 0.2)
                idxs_to_restart = np.random.choice(self.pop_size, num_to_restart, replace=False)
                self.population[idxs_to_restart] = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(num_to_restart, self.dim))
                new_fitness = np.array([func(x) for x in self.population[idxs_to_restart]])
                self.budget -= num_to_restart
                fitness[idxs_to_restart] = new_fitness
                
                for i in idxs_to_restart:
                    if fitness[i] < self.f_opt:
                        self.f_opt = fitness[i]
                        self.x_opt = self.population[i]

            self.best_fitness_history.append(self.f_opt)

        return self.f_opt, self.x_opt

    def generate_orthogonal_array(self, n):
        # Simplified orthogonal array generation (Hadamard matrix based)
        # Not a complete implementation for all n, but works for powers of 2
        n = int(2**np.ceil(np.log2(n)))  # Pad to next power of 2
        if n == 1:
            return np.array([[1]])
        H = np.array([[1]])
        while H.shape[0] < n:
            H = np.vstack((np.hstack((H, H)), np.hstack((H, -H))))
        return H[:n]