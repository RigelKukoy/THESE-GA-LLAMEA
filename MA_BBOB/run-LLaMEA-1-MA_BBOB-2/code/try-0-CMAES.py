import numpy as np

class CMAES:
    def __init__(self, budget=10000, dim=10, mu_ratio=0.25, cs=0.3, dsigma=0.2, ccov=0.3):
        self.budget = budget
        self.dim = dim
        self.mu_ratio = mu_ratio
        self.cs = cs
        self.dsigma = dsigma
        self.ccov = ccov
        self.lb = -5.0
        self.ub = 5.0
        

    def __call__(self, func):
        # Initialize variables
        
        xmean = np.random.uniform(self.lb, self.ub, size=self.dim)  # Initial guess of mean
        sigma = 0.5  # Overall step size
        
        lambda_ = int(4 + np.floor(3 * np.log(self.dim)))  # Population size
        mu = int(lambda_ * self.mu_ratio)  # Number of parents
        
        weights = np.log(mu + 1/2) - np.log(np.arange(1, mu + 1))  # Weights for recombination
        weights = weights / np.sum(weights)
        
        mueff = np.sum(weights)**2 / np.sum(weights**2)
        
        C = np.eye(self.dim)  # Covariance matrix
        pc = np.zeros(self.dim)  # Evolution path for C
        ps = np.zeros(self.dim)  # Evolution path for sigma
        
        chiN = np.sqrt(self.dim) * (1 - (1/(4*self.dim)) + 1/(12*self.dim**2))
        
        # Parameters for adaption
        cs = self.cs  # Step-size damping
        damps = 1 + self.dsigma * max(0, np.sqrt((mueff-1)/(self.dim+1))-1) + cs  # Damping for step-size
        ccov = self.ccov
        c1 = ccov / ((self.dim+1.3)**2 + mueff)
        cmu = min(1-c1, ccov * (mueff-2+1/mueff) / ((self.dim+2.0)**2 + mueff))
        
        f_opt = np.Inf
        x_opt = None
        evals = 0

        restart_iter = 0
        max_restarts = 3

        while evals < self.budget:
            # Generate and evaluate offspring
            z = np.random.normal(0, 1, size=(self.dim, lambda_))
            y = np.dot(np.linalg.cholesky(C), z)
            x = xmean.reshape(-1, 1) + sigma * y
            x = np.clip(x, self.lb, self.ub)
            
            f = np.array([func(x[:,i]) if evals + i < self.budget else np.inf for i in range(lambda_)])
            evals += lambda_
            
            # Sort by fitness
            idx = np.argsort(f)
            f = f[idx]
            x = x[:, idx]
            
            # Update optimal solution
            if f[0] < f_opt:
                f_opt = f[0]
                x_opt = x[:, 0]
            
            # Update distribution parameters
            xmean_new = np.sum(x[:, :mu] * weights, axis=1)
            
            ps = (1-cs) * ps + np.sqrt(cs*(2-cs)*mueff) * np.dot(np.linalg.inv(np.linalg.cholesky(C)), (xmean_new - xmean)) / sigma
            hsig = np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*evals/lambda_))/chiN < 1.4 + 2/(self.dim+1)
            pc = (1-ccov) * pc + hsig * np.sqrt(ccov*(2-ccov)*mueff) * (xmean_new - xmean) / sigma
            
            C = (1-c1-cmu) * C + c1 * (np.outer(pc, pc) + (1-hsig) * ccov*(2-ccov) * C)
            C += cmu * np.sum(weights[:, None, None] * (np.transpose(x[:, :mu] - xmean.reshape(-1,1)) * (x[:, :mu] - xmean.reshape(-1,1))), axis=0) / sigma**2
            
            sigma = sigma * np.exp((cs/damps) * (np.linalg.norm(ps)/chiN - 1))
            
            xmean = xmean_new

            # Repair covariance matrix (ensure positive definiteness)
            C = np.triu(C) + np.transpose(np.triu(C,1))
            try:
                np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = C + np.eye(self.dim) * 1e-6
            
            # Restart mechanism
            if np.max(np.diag(C)) > (10**7) * sigma:
                restart_iter += 1
                xmean = np.random.uniform(self.lb, self.ub, size=self.dim)
                sigma = 0.5
                C = np.eye(self.dim)
                pc = np.zeros(self.dim)
                ps = np.zeros(self.dim)
                if restart_iter > max_restarts:
                    break
                    
        return f_opt, x_opt