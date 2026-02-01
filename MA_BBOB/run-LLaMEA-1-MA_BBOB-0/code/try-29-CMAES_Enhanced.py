import numpy as np

class CMAES_Enhanced:
    def __init__(self, budget=10000, dim=10, popsize=None, cs=0.3, damps=None, c_cov=None, mu_factor=0.25, sigma0=0.5, memory_size=100, restart_trigger=1e-12, orthogonal_sampling=True, learning_rate_decay=0.99):
        self.budget = budget
        self.dim = dim
        self.popsize = popsize if popsize is not None else 4 + int(3 * np.log(self.dim))
        self.mu = int(self.popsize * mu_factor)
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.m = np.zeros(self.dim)
        self.sigma = sigma0
        self.ps = np.zeros(self.dim)
        self.pc = np.zeros(self.dim)
        self.C = np.eye(self.dim)
        self.cs = cs
        self.damps = damps if damps is not None else 1 + 2 * np.max((0, np.sqrt((self.mu / self.dim) - 1))) + self.cs
        self.c_cov = c_cov if c_cov is not None else (1 / self.mu) * (2 / ((self.dim + np.sqrt(2))**2))
        self.c_cov_mu = 0.25 + self.c_cov
        self.chiN = self.dim**0.5 * (1 - (1 / (4 * self.dim)) + (1 / (21 * self.dim**2)))
        self.f_opt = np.Inf
        self.x_opt = None
        self.archive = []
        self.archive_size = memory_size
        self.evals = 0
        self.restart_trigger = restart_trigger
        self.orthogonal_sampling = orthogonal_sampling
        self.successful_sigma_updates = []
        self.learning_rate_decay = learning_rate_decay
        self.eigenvalues = None # Store eigenvalues for spectral initialization
        self.last_f_opt = np.Inf
        self.stagnation_counter = 0
        self.stagnation_threshold = 10 # Number of iterations without improvement before considering restart
        self.min_sigma = 1e-16 # Minimum allowed sigma value


    def __call__(self, func):
        self.evals = 0
        self.successful_sigma_updates = []
        self.last_f_opt = np.Inf
        self.stagnation_counter = 0
        self._spectral_initialization(func) # Initialize covariance matrix based on function landscape
        
        while self.evals < self.budget:
            # Adaptive Population Size
            self.popsize = 4 + int(3 * np.log(self.dim))
            self.mu = int(self.popsize * 0.25)
            self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
            self.weights /= np.sum(self.weights)
            
            # Sample population
            if self.orthogonal_sampling:
                z = self._orthogonal_sampling(self.popsize, self.dim)
            else:
                z = np.random.normal(0, 1, size=(self.popsize, self.dim))
            C_sqrt = np.linalg.cholesky(self.C)
            x = self.m + self.sigma * z @ C_sqrt.T
            
            # Clip to bounds.
            lb = func.bounds.lb
            ub = func.bounds.ub
            x = np.clip(x, lb, ub)
            
            f = np.array([func(xi) for xi in x])
            self.evals += self.popsize
            
            # Sort by fitness
            idx = np.argsort(f)
            x = x[idx]
            f = f[idx]
            
            if f[0] < self.f_opt:
                self.f_opt = f[0]
                self.x_opt = x[0]
                self.stagnation_counter = 0  # Reset stagnation counter
            else:
                self.stagnation_counter += 1

            # Update mean
            m_old = self.m.copy()
            self.m = np.sum(self.weights[:, None] * x[:self.mu], axis=0)

            # Update evolution paths
            self.ps = (1 - self.cs) * self.ps + (np.sqrt(self.cs * (2 - self.cs) * self.weights @ self.weights)) * (self.m - m_old) @ np.linalg.inv(C_sqrt).T
            self.pc = (1 - self.c_cov) * self.pc + np.sqrt(self.c_cov * (2 - self.c_cov)) * (self.m - m_old) / self.sigma

            # Update covariance matrix
            hsigma = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs)**(2 * (self.evals / self.popsize))) < (1.4 + (2 / (self.dim + 1))) * self.chiN
            
            dC = (self.c_cov_mu / self.mu) * (self.pc[:, None] @ self.pc[None, :])
            for i in range(self.mu):
                dC += (self.c_cov / self.mu) * ((x[i] - m_old)[:, None] @ (x[i] - m_old)[None, :]) / self.sigma**2
                
            self.C = (1 - self.c_cov - self.c_cov_mu) * self.C + dC
            
            # Ensure positive definiteness
            try:
                np.linalg.cholesky(self.C)
            except np.linalg.LinAlgError:
                self.C = np.eye(self.dim)
                

            # Adaptive step size control
            old_sigma = self.sigma
            self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))
            self.sigma = max(self.sigma, self.min_sigma) #Prevent sigma from becoming too small

            # Store successful sigma updates
            if self.f_opt < np.inf: # Check if a better solution has been found
                self.successful_sigma_updates.append((old_sigma, self.sigma))

            # Weighted average of successful step sizes
            if len(self.successful_sigma_updates) > 5:
                 weights = np.exp(-np.arange(len(self.successful_sigma_updates)) / 2.0)
                 weights /= np.sum(weights)
                 weighted_sigma = np.sum([w * s[1] for w, s in zip(weights[::-1], self.successful_sigma_updates[-len(weights):])])
                 self.sigma = weighted_sigma


            # Archive for diversity (optional)
            for xi in x:
                self.archive.append(xi)
            
            # Dynamic archive size
            self.archive_size = min(int(self.budget / 10), 100 + int(self.evals / self.budget * 200))
            if len(self.archive) > self.archive_size:
                self.archive = self.archive[-self.archive_size:]
            
            #Enhanced Restart mechanism: Stagnation and Fitness Improvement
            if self.stagnation_counter > self.stagnation_threshold or self.sigma < self.restart_trigger:
                if self.f_opt < self.last_f_opt: # Significant improvement since last restart
                    self.last_f_opt = self.f_opt
                    self.m = self.x_opt.copy() #Keep best solution
                else:
                    self.m = np.mean(np.array(self.archive), axis=0) if self.archive else np.zeros(self.dim) #Restart from archive

                self.sigma = self.sigma0 * (self.learning_rate_decay**(self.evals / self.budget)) #Decay sigma
                self.ps = np.zeros(self.dim)
                self.pc = np.zeros(self.dim)
                self.C = np.eye(self.dim)
                self.stagnation_counter = 0


        return self.f_opt, self.x_opt

    def _orthogonal_sampling(self, popsize, dim):
        # Generate standard normal samples
        z = np.random.normal(0, 1, size=(popsize, dim))

        # Perform QR decomposition
        Q, R = np.linalg.qr(z)

        # Normalize columns to have unit length
        for i in range(popsize):
            Q[i, :] /= np.linalg.norm(Q[i, :])

        return Q
    
    def _spectral_initialization(self, func):
        # Sample a set of points
        num_samples = min(1000, self.budget // 10)
        X = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(num_samples, self.dim))
        F = np.array([func(x) for x in X])

        # Compute the gradient at each point (approximate using finite differences)
        gradients = np.zeros((num_samples, self.dim))
        h = 1e-5 #Step size for finite differences
        for i in range(num_samples):
            for j in range(self.dim):
                x_plus_h = X[i].copy()
                x_plus_h[j] += h
                x_plus_h = np.clip(x_plus_h, func.bounds.lb, func.bounds.ub)
                gradients[i, j] = (func(x_plus_h) - F[i]) / h

        # Compute the covariance matrix of the gradients
        try:
            grad_cov = np.cov(gradients, rowvar=False)
        except:
            grad_cov = np.eye(self.dim)

        # Eigendecomposition
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(grad_cov)
        except:
             eigenvalues = np.ones(self.dim)
             eigenvectors = np.eye(self.dim)

        #Sort eigenvalues and eigenvectors
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Normalize eigenvalues to have a sum equal to dim (trace of covariance matrix)
        eigenvalues = (eigenvalues / np.sum(eigenvalues)) * self.dim
        self.eigenvalues = eigenvalues

        # Reconstruct the covariance matrix
        self.C = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        
        #Ensure positive definiteness
        try:
            np.linalg.cholesky(self.C)
        except np.linalg.LinAlgError:
            self.C = np.eye(self.dim)