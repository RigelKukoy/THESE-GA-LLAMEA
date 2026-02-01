import numpy as np

class AdaptiveCMAES:
    def __init__(self, budget=10000, dim=10, mu_ratio=0.25, cs=0.3, dsigma=0.2, ccov=0.3, initial_sigma=0.5, active=True, orthogonal_learning=True):
        self.budget = budget
        self.dim = dim
        self.mu_ratio = mu_ratio
        self.cs = cs
        self.dsigma = dsigma
        self.ccov = ccov
        self.lb = -5.0
        self.ub = 5.0
        self.initial_sigma = initial_sigma
        self.active = active
        self.orthogonal_learning = orthogonal_learning

    def __call__(self, func):
        # Initialize variables
        xmean = np.random.uniform(self.lb, self.ub, size=self.dim)
        sigma = self.initial_sigma

        lambda_ = int(4 + np.floor(3 * np.log(self.dim)))  # Initial population size
        mu = int(lambda_ * self.mu_ratio)
        
        weights = np.log(mu + 1/2) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        
        mueff = np.sum(weights)**2 / np.sum(weights**2)
        
        C = np.eye(self.dim)
        pc = np.zeros(self.dim)
        ps = np.zeros(self.dim)
        
        chiN = np.sqrt(self.dim) * (1 - (1/(4*self.dim)) + 1/(12*self.dim**2))
        
        # Parameters for adaption
        cs = self.cs
        damps = 1 + self.dsigma * max(0, np.sqrt((mueff-1)/(self.dim+1))-1) + cs
        ccov = self.ccov
        c1 = ccov / ((self.dim+1.3)**2 + mueff)
        cmu = min(1-c1, ccov * (mueff-2+1/mueff) / ((self.dim+2.0)**2 + mueff))

        # Active CMA parameters
        c1a = c1
        cmua = cmu
        if self.active:
            c1a = c1 / 10
            cmua = min(1 - c1a, cmu * (mueff - 2 + 1/mueff) / ((self.dim + 2)**2 + mueff))
        
        f_opt = np.Inf
        x_opt = None
        evals = 0

        restart_iter = 0
        max_restarts = 5

        stagnation_counter = 0
        stagnation_threshold = 50

        condition_number_threshold = 1e6

        # Adaptive population sizing
        lambda_min = 4 + np.floor(3 * np.log(self.dim))
        lambda_max = 2 * lambda_min
        
        # Orthogonal subspace learning parameters
        if self.orthogonal_learning:
            orth_ subspace_dim = min(self.dim // 2, mu)  # Dimension of the subspace
            B = np.eye(self.dim)

        while evals < self.budget:
            # Dynamically adjust population size based on performance
            if stagnation_counter > stagnation_threshold // 2:
                lambda_ = min(lambda_ + 2, lambda_max)  # Increase population size
            else:
                lambda_ = max(lambda_ - 1, lambda_min) # Reduce population size if performing well.
            mu = int(lambda_ * self.mu_ratio)

            # Generate and evaluate offspring
            z = np.random.normal(0, 1, size=(self.dim, lambda_))
            y = np.dot(np.linalg.cholesky(C), z)
            x = xmean.reshape(-1, 1) + sigma * y
            x = np.clip(x, self.lb, self.ub)

            # Mirrored sampling
            x_mirrored = xmean.reshape(-1, 1) - sigma * y
            x_mirrored = np.clip(x_mirrored, self.lb, self.ub)
            x_combined = np.concatenate((x, x_mirrored), axis=1)
            
            f = np.array([func(x_combined[:,i]) if evals + i < self.budget else np.inf for i in range(2 * lambda_)])
            evals += 2 * lambda_

            # Sort by fitness
            idx = np.argsort(f)
            f = f[idx]
            x_combined = x_combined[:, idx]
            x = x_combined[:, :lambda_] # Select only lambda best
            f = f[:lambda_]
            
            # Update optimal solution
            if f[0] < f_opt:
                f_opt = f[0]
                x_opt = x[:, 0].copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Update distribution parameters
            xmean_new = np.sum(x[:, :mu] * weights, axis=1)
            
            ps = (1-cs) * ps + np.sqrt(cs*(2-cs)*mueff) * np.dot(np.linalg.inv(np.linalg.cholesky(C)), (xmean_new - xmean)) / sigma
            hsig = np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*evals/lambda_))/chiN < 1.4 + 2/(self.dim+1)
            pc = (1-ccov) * pc + hsig * np.sqrt(ccov*(2-ccov)*mueff) * (xmean_new - xmean) / sigma
            
            C = (1-c1-cmu) * C + c1 * (np.outer(pc, pc) + (1-hsig) * ccov*(2-ccov) * C)
            
            # More robust covariance update, selective update
            for i in range(mu):
                y = (x[:, i] - xmean) / sigma
                C += cmu * weights[i] * np.outer(y, y)

            # Active covariance update
            if self.active:
                for i in range(mu, min(lambda_, 2*mu)):
                    w = weights[-1] / 10
                    y = (x[:, i] - xmean) / sigma
                    C -= cmua * w * np.outer(y, y)

            # Step-size adaptation with composite strategy
            sigma_old = sigma
            sigma = sigma * np.exp((cs/damps) * (np.linalg.norm(ps)/chiN - 1))
            if stagnation_counter > stagnation_threshold // 2: # Composite adaptation
                sigma = 0.5 * sigma + 0.5 * sigma_old
            
            xmean = xmean_new

             # Orthogonal subspace learning
            if self.orthogonal_learning:
                # Select the best mu individuals
                x_best = x[:, :mu]
                # Center the data
                X = x_best - xmean.reshape(-1, 1)

                # Perform SVD
                U, S, V = np.linalg.svd(X)

                # Select the orthogonal subspace
                B = U[:, :orth_ subspace_dim]

                # Learn within the orthogonal subspace
                z_orth = np.random.normal(0, 1, size=(orth_ subspace_dim, lambda_))
                y_orth = np.dot(B, z_orth)
                x_orth = xmean.reshape(-1, 1) + sigma * y_orth
                x_orth = np.clip(x_orth, self.lb, self.ub)

                f_orth = np.array([func(x_orth[:, i]) if evals + i < self.budget else np.inf for i in range(lambda_)])
                evals += lambda_

                idx_orth = np.argsort(f_orth)
                f_orth = f_orth[idx_orth]
                x_orth = x_orth[:, idx_orth]

                if f_orth[0] < f_opt:
                    f_opt = f_orth[0]
                    x_opt = x_orth[:, 0].copy()
                    stagnation_counter = 0

            # Repair covariance matrix
            C = np.triu(C) + np.transpose(np.triu(C,1))
            try:
                np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = C + np.eye(self.dim) * 1e-6
            
            # Adaptive Restart mechanism
            if stagnation_counter > stagnation_threshold:
                eigenvalues = np.linalg.eigvalsh(C)
                condition_number = np.max(eigenvalues) / np.min(eigenvalues)

                if condition_number > condition_number_threshold:
                    restart_iter += 1
                    xmean = np.random.uniform(self.lb, self.ub, size=self.dim)
                    sigma = self.initial_sigma
                    C = np.eye(self.dim)
                    pc = np.zeros(self.dim)
                    ps = np.zeros(self.dim)
                    stagnation_counter = 0

                    condition_number_threshold *= 0.8

                    if restart_iter > max_restarts:
                        break
                else:
                     stagnation_counter = 0

            if np.any(np.isnan(C)):
                C = np.eye(self.dim)
                sigma = self.initial_sigma
                pc = np.zeros(self.dim)
                ps = np.zeros(self.dim)
                stagnation_counter = 0
                    
        return f_opt, x_opt