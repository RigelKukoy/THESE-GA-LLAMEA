import numpy as np

class AdaptiveODE:
    def __init__(self, budget=10000, dim=10, pop_size=50, archive_size=10, pbest_rate=0.1, F=0.5, CR=0.7, F_lr=0.1, CR_lr=0.1, restart_trigger=50):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.archive_size = archive_size
        self.pbest_rate = pbest_rate
        self.F = F
        self.CR = CR
        self.F_lr = F_lr
        self.CR_lr = CR_lr
        self.population = None
        self.fitness = None
        self.best_index = None
        self.archive = None
        self.archive_fitness = None
        self.f_opt = np.inf
        self.x_opt = None
        self.success_history_F = []
        self.success_history_CR = []
        self.restart_trigger = restart_trigger
        self.no_improvement_count = 0

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        self.best_index = np.argmin(self.fitness)
        self.f_opt = self.fitness[self.best_index]
        self.x_opt = self.population[self.best_index].copy()
        self.archive = np.zeros((self.archive_size, self.dim))
        self.archive_fitness = np.full(self.archive_size, np.inf)

    def current_to_pbest_mutation(self, x, pbest_indices, x_r1, F):
        x_pbest = self.population[np.random.choice(pbest_indices)]
        return x + F * (x_pbest - x) + F * (x_r1 - x)
    
    def update_archive(self, x, f_x):
        if np.any(f_x < self.archive_fitness):
            worst_index = np.argmax(self.archive_fitness)
            self.archive[worst_index] = x
            self.archive_fitness[worst_index] = f_x

    def orthogonal_learning(self, x, func):
        basis = np.random.randn(self.dim, self.dim)
        basis, _ = np.linalg.qr(basis)
        
        search_range = 0.1 * (func.bounds.ub - func.bounds.lb)
        coefficients = np.random.uniform(-search_range, search_range, size=self.dim)
        
        best_x = x.copy()
        best_f = func(x)
        self.budget -= 1
        
        for i in range(self.dim):
            new_x = x + coefficients[i] * basis[:, i]
            new_x = np.clip(new_x, func.bounds.lb, func.bounds.ub)
            new_f = func(new_x)
            self.budget -= 1
            if new_f < best_f:
                best_f = new_f
                best_x = new_x.copy()

        return best_x, best_f
    
    def restart_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        self.best_index = np.argmin(self.fitness)
        self.f_opt = self.fitness[self.best_index]
        self.x_opt = self.population[self.best_index].copy()

    def __call__(self, func):
        self.initialize_population(func)

        while self.budget > 0:
            for i in range(self.pop_size):
                # Parameter adaptation using success history
                if self.success_history_F:
                    self.F = np.random.choice(self.success_history_F)
                if self.success_history_CR:
                    self.CR = np.random.choice(self.success_history_CR)
                    
                # Mutation
                pbest_count = max(1, int(self.pbest_rate * self.pop_size))
                pbest_indices = np.argsort(self.fitness)[:pbest_count]
                
                indices = np.random.choice(self.pop_size, 1, replace=False)
                x_r1 = self.population[indices[0]]
                
                v = self.current_to_pbest_mutation(self.population[i], pbest_indices, x_r1, self.F)
                v = np.clip(v, func.bounds.lb, func.bounds.ub)
                
                # Crossover
                j_rand = np.random.randint(self.dim)
                u = self.population[i].copy()
                for j in range(self.dim):
                    if np.random.rand() < self.CR or j == j_rand:
                        u[j] = v[j]

                # Evaluation
                f_u = func(u)
                self.budget -= 1

                # Selection
                if f_u < self.fitness[i]:
                    self.success_history_F.append(self.F)
                    self.success_history_CR.append(self.CR)
                    
                    self.fitness[i] = f_u
                    self.population[i] = u
                    
                    if f_u < self.f_opt:
                        self.f_opt = f_u
                        self.x_opt = u.copy()
                        self.no_improvement_count = 0
                        
                    self.update_archive(u, f_u)
                else:
                    self.no_improvement_count += 1

                # Limit the size of the success history
                self.success_history_F = self.success_history_F[-10:]
                self.success_history_CR = self.success_history_CR[-10:]

            # Orthogonal learning around the best solution
            best_x, best_f = self.orthogonal_learning(self.x_opt, func)
            if best_f < self.f_opt:
                self.f_opt = best_f
                self.x_opt = best_x.copy()
                self.no_improvement_count = 0

            if self.no_improvement_count > self.restart_trigger:
                self.restart_population(func)
                self.no_improvement_count = 0

        return self.f_opt, self.x_opt