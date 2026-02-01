import numpy as np

class CMAES_CauchyDE:
    def __init__(self, budget=10000, dim=10, popsize=None, F=0.5, CR=0.7, initial_sigma=0.1, cs=0.3, cc=0.1, mu_ratio=0.25):
        self.budget = budget
        self.dim = dim
        self.popsize = popsize if popsize is not None else 4 + int(3 * np.log(self.dim)) # Recommended CMA-ES popsize
        self.F = F
        self.CR = CR
        self.initial_sigma = initial_sigma
        self.sigma = self.initial_sigma
        self.cs = cs
        self.cc = cc
        self.mu = int(self.popsize * mu_ratio)
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.m = None
        self.C = None
        self.pc = None
        self.ps = None
        self.chiN = None
        self.eval_count = 0


    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub

        # Initialization
        self.m = np.random.uniform(lb, ub, size=self.dim)
        self.C = np.eye(self.dim)
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.chiN = np.sqrt(self.dim) * (1 - 1/(4*self.dim) + 1/(21*self.dim**2))

        self.f_opt = np.Inf
        self.x_opt = None
        
        while self.eval_count < self.budget:
            # Generate population
            z = np.random.randn(self.dim, self.popsize)
            x = self.m[:, np.newaxis] + self.sigma * np.dot(np.linalg.cholesky(self.C), z)
            x = np.clip(x, lb, ub)
            
            fitness = np.array([func(xi) for xi in x.T])
            self.eval_count += self.popsize

            # Sort by fitness
            idx = np.argsort(fitness)
            fitness = fitness[idx]
            x = x[:, idx]
            
            # Update optimal solution
            if fitness[0] < self.f_opt:
                self.f_opt = fitness[0]
                self.x_opt = x[:, 0]

            # Update CMA-ES parameters
            m_old = self.m.copy()
            self.m = np.sum(x[:, :self.mu] * self.weights[np.newaxis, :], axis=1)
            
            self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs)) * np.linalg.solve(np.linalg.cholesky(self.C), (self.m - m_old)) / self.sigma
            self.pc = (1 - self.cc) * self.pc + np.sqrt(self.cc * (2 - self.cc)) * (self.m - m_old) / self.sigma
            
            hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs)**(2 * self.eval_count / self.popsize)) < self.chiN * (1.4 + 2 / (self.dim + 1))
            
            self.C = (1 - self.cc) * self.C + self.cc * (1 / np.min([1, hsig**2 + 0.3])) * (np.outer(self.pc, self.pc))
            for i in range(self.mu):
                y = (x[:, i] - m_old) / self.sigma
                self.C += self.weights[i] * np.outer(y, y)
            
            self.sigma *= np.exp((self.cs / 2) * (np.linalg.norm(self.ps) / self.chiN - 1))

            # Cauchy mutation to enhance exploration
            cauchy_scale = self.F * (ub - lb)
            for i in range(self.popsize):
                if np.random.rand() < 0.1: # Apply Cauchy with probability 0.1
                  z_cauchy = np.random.standard_cauchy(size=self.dim)
                  x_cauchy = self.m + cauchy_scale * z_cauchy
                  x_cauchy = np.clip(x_cauchy, lb, ub)
                  f_cauchy = func(x_cauchy)
                  self.eval_count += 1

                  if f_cauchy < fitness[i]:
                      x[:, i] = x_cauchy
                      fitness[i] = f_cauchy
                      if fitness[i] < self.f_opt:
                          self.f_opt = fitness[i]
                          self.x_opt = x_cauchy

            if self.eval_count >= self.budget:
                break

        return self.f_opt, self.x_opt