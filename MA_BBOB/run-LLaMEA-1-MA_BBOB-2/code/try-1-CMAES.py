import numpy as np

class CMAES:
    def __init__(self, budget=10000, dim=10, popsize=None, cs=0.3, damps=1, ccov1=0.0, ccovmu=0.0, sigma0=0.2):
        self.budget = budget
        self.dim = dim
        self.popsize = popsize if popsize is not None else 4 + int(3 * np.log(self.dim))
        self.mu = self.popsize // 2
        self.weights = np.log(self.mu + 1/2) - np.log(np.arange(1, self.mu + 1))
        self.weights = self.weights / np.sum(self.weights)
        self.m = None  # Mean
        self.sigma = sigma0  # Step size
        self.C = None  # Covariance matrix
        self.pc = None # Evolution path for covariance matrix
        self.ps = None # Evolution path for step size
        self.cs = cs  # Step-size damping
        self.damps = damps # Damping for step-size
        self.ccov1 = ccov1  # Learning rate for rank-one update
        self.ccovmu = ccovmu # Learning rate for rank-mu update
        self.eigenspace_initialized = False
        self.B = None  # Matrix of eigenvectors
        self.D = None  # Vector of eigenvalues
        self.invsqrtC = None
        self.x_opt = None
        self.f_opt = np.Inf

    def __call__(self, func):
        self.m = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
        self.C = np.eye(self.dim)
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        
        self.ccov1 = 2 / ((self.dim + 1.3)**2 + self.mu)
        self.ccovmu = min(1 - self.ccov1, 2 * (self.mu - 1 + 1/self.mu) / ((self.dim + 2)**2 + 2*self.mu))
        self.cs = (self.damps*(self.mu_eff/np.linalg.norm(self.ps)) < 1)
        self.damps = 1 + 2*max(0, np.sqrt((self.mu_eff-1)/(self.dim+1)) -1) + self.cs

        used_budget = 0
        while used_budget < self.budget:
            # Generate samples
            z = np.random.multivariate_normal(np.zeros(self.dim), np.eye(self.dim), size=self.popsize)
            x = self.m + self.sigma * (self.B @ (self.D * z.T)).T
            
            # Repair solutions
            x = np.clip(x, func.bounds.lb, func.bounds.ub)

            # Evaluate samples
            f = np.array([func(xi) for xi in x])
            used_budget += self.popsize

            # Sort by fitness
            idx = np.argsort(f)
            x = x[idx]
            f = f[idx]
            
            # Update optimal solution
            if f[0] < self.f_opt:
                self.f_opt = f[0]
                self.x_opt = x[0]

            # Selection and recombination
            x_mu = x[:self.mu]
            z_mu = z[idx[:self.mu]]
            
            delta_m = np.sum(self.weights.reshape(-1, 1) * z_mu, axis=0)
            self.m = self.m + self.sigma * self.B @ (self.D * delta_m)

            # Update evolution paths
            self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * (self.invsqrtC @ delta_m)
            hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs)**(2 * (used_budget / self.popsize))) < (1.4 + 2 / (self.dim + 1))

            self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * delta_m
            
            # Update covariance matrix
            rankone = np.outer(self.pc, self.pc)
            rankmu = np.sum(self.weights.reshape(-1, 1, 1) * np.array([np.outer(z_mu[i], z_mu[i]) for i in range(self.mu)]), axis=0)
            self.C = (1 - self.ccov1 - self.ccovmu + self.ccov1 * self.cc * (2 - self.cc)) * self.C + self.ccov1 * rankone + self.ccovmu * rankmu

            # Update step size
            self.sigma = self.sigma * np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / np.sqrt(self.dim) - 1))
            
            # Eigen decomposition
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            D, B = np.linalg.eigh(self.C)

            self.D = np.sqrt(D)
            self.B = B
            self.invsqrtC = self.B @ np.diag(self.D**(-1.0)) @ self.B.T

        return self.f_opt, self.x_opt

    @property
    def mu_eff(self):
        return np.sum(self.weights)**2 / np.sum(self.weights**2)

    @property
    def cc(self):
        return (4 + self.mu_eff/self.dim) / (self.dim + 4 + 2*self.mu_eff/self.dim)