import numpy as np

class SaCmaLocalDE:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.5, CR=0.7, local_search_prob=0.1, ls_radius=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.local_search_prob = local_search_prob
        self.ls_radius = ls_radius
        self.population = None
        self.fitness = None
        self.best_index = None
        self.f_opt = np.inf
        self.x_opt = None
        self.covariance_matrix = None
        self.mean = None


    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.budget -= self.pop_size
        self.best_index = np.argmin(self.fitness)
        self.f_opt = self.fitness[self.best_index]
        self.x_opt = self.population[self.best_index].copy()
        self.mean = np.mean(self.population, axis=0)
        self.covariance_matrix = np.cov(self.population, rowvar=False)
        if np.sum(np.isnan(self.covariance_matrix)) > 0:
            self.covariance_matrix = np.eye(self.dim) * 0.01 # fallback


    def mutation(self, i):
        indices = np.random.choice(self.pop_size, 3, replace=False)
        x_r1, x_r2, x_r3 = self.population[indices]

        # Self-adaptive F
        F = np.random.normal(self.F, 0.1)
        F = np.clip(F, 0.1, 1.0)

        v = x_r1 + F * (x_r2 - x_r3)
        return v


    def crossover(self, mutant, target):
         # CMA-ES like crossover
        u = target.copy()
        j_rand = np.random.randint(self.dim)
        
        # Ensure covariance matrix is positive semi-definite by adding a small constant to the diagonal
        covariance_matrix = self.covariance_matrix + np.eye(self.dim) * 1e-6
        
        try:
          L = np.linalg.cholesky(covariance_matrix)
          z = np.random.normal(0, 1, size=self.dim)
          offspring = self.mean + np.dot(L, z)
          offspring = np.clip(offspring, -5.0, 5.0)
        
          for j in range(self.dim):
              if np.random.rand() < self.CR or j == j_rand:
                u[j] = mutant[j] # mutant[j]
              else:
                u[j] = offspring[j]
        except np.linalg.LinAlgError:
          for j in range(self.dim):
              if np.random.rand() < self.CR or j == j_rand:
                  u[j] = mutant[j]
        return u


    def local_search(self, x, func):
        x_new = x + np.random.uniform(-self.ls_radius, self.ls_radius, size=self.dim)
        x_new = np.clip(x_new, func.bounds.lb, func.bounds.ub)
        f_new = func(x_new)
        self.budget -= 1
        return x_new, f_new


    def __call__(self, func):
        self.initialize_population(func)

        while self.budget > 0:
            for i in range(self.pop_size):
                # Mutation
                v = self.mutation(i)
                v = np.clip(v, func.bounds.lb, func.bounds.ub)

                # Crossover
                u = self.crossover(v, self.population[i])
                u = np.clip(u, func.bounds.lb, func.bounds.ub)

                # Evaluation
                f_u = func(u)
                self.budget -= 1

                # Selection
                if f_u < self.fitness[i]:
                    self.fitness[i] = f_u
                    self.population[i] = u
                    if f_u < self.f_opt:
                        self.f_opt = f_u
                        self.x_opt = u.copy()

                # Local search
                if np.random.rand() < self.local_search_prob and self.budget > 0:
                    x_ls, f_ls = self.local_search(self.population[i], func)
                    if f_ls < self.fitness[i]:
                        self.fitness[i] = f_ls
                        self.population[i] = x_ls
                        if f_ls < self.f_opt:
                            self.f_opt = f_ls
                            self.x_opt = x_ls.copy()
            
            self.mean = np.mean(self.population, axis=0)
            self.covariance_matrix = np.cov(self.population, rowvar=False)
            if np.sum(np.isnan(self.covariance_matrix)) > 0:
                self.covariance_matrix = np.eye(self.dim) * 0.01
            

        return self.f_opt, self.x_opt