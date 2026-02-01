import numpy as np

class CMAES:
    def __init__(self, budget=10000, dim=10, mu_ratio=0.25, cs=0.3, dsigma=0.2, ccov=0.3, initial_sigma=0.5, exploration_factor=1.0):
        self.budget = budget
        self.dim = dim
        self.mu_ratio = mu_ratio
        self.cs = cs
        self.dsigma = dsigma
        self.ccov = ccov
        self.lb = -5.0
        self.ub = 5.0
        self.initial_sigma = initial_sigma
        self.exploration_factor = exploration_factor # Exploration factor

    def __call__(self, func):
        # Initialize variables
        xmean = np.random.uniform(self.lb, self.ub, size=self.dim)
        sigma = self.initial_sigma
        lambda_ = int(4 + np.floor(3 * np.log(self.dim)))
        mu = int(lambda_ * self.mu_ratio)
        weights = np.log(mu + 1/2) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = np.sum(weights)**2 / np.sum(weights**2)
        C = np.eye(self.dim)
        pc = np.zeros(self.dim)
        ps = np.zeros(self.dim)
        chiN = np.sqrt(self.dim) * (1 - (1/(4*self.dim)) + 1/(12*self.dim**2))
        cs = self.cs
        damps = 1 + self.dsigma * max(0, np.sqrt((mueff-1)/(self.dim+1))-1) + cs
        ccov = self.ccov
        c1 = ccov / ((self.dim+1.3)**2 + mueff)
        cmu = min(1-c1, ccov * (mueff-2+1/mueff) / ((self.dim+2.0)**2 + mueff))
        f_opt = np.Inf
        x_opt = None
        evals = 0
        restart_iter = 0
        max_restarts = 3
        stagnation_counter = 0
        previous_f_opt = np.Inf
        stagnation_threshold = 1e-9
        stagnation_patience = 50

        while evals < self.budget:
            # Dynamic population size adjustment
            lambda_ = int(4 + np.floor(3 * np.log(self.dim) * (1 - evals/self.budget)))
            mu = int(lambda_ * self.mu_ratio)
            weights = np.log(mu + 1/2) - np.log(np.arange(1, mu + 1))
            weights = weights / np.sum(weights)
            mueff = np.sum(weights)**2 / np.sum(weights**2)

            # Generate and evaluate offspring (Orthogonal Sampling)
            z = np.random.normal(0, 1, size=(self.dim, lambda_))

            # Increased exploration with orthogonal sampling and exploration factor
            Q, _ = np.linalg.qr(z)
            y = np.dot(np.linalg.cholesky(C), Q) * (1 + self.exploration_factor * np.random.rand())  # Apply Exploration factor
            x = xmean.reshape(-1, 1) + sigma * y

            # Bounds penalty
            x_clipped = np.clip(x, self.lb, self.ub)
            penalty = np.sum((x - x_clipped)**2, axis=0)
            x = x_clipped
            f = np.array([func(x[:,i]) + 1e3 * penalty[i] if evals + i < self.budget else np.inf for i in range(lambda_)])
            evals += lambda_

            # Sort by fitness
            idx = np.argsort(f)
            f = f[idx]
            x = x[:, idx]

            # Selective pressure adjustment (increased selection pressure in later stages)
            mu_adjusted = int(mu * (0.5 + 0.5 * (1 - evals / self.budget)))
            if mu_adjusted < 1:
                mu_adjusted = 1
            
            # Update optimal solution
            if f[0] < f_opt:
                f_opt = f[0]
                x_opt = x[:, 0].copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Stagnation Check (Improved)
            if abs(f_opt - previous_f_opt) < stagnation_threshold:
                stagnation_counter += 1
            else:
                stagnation_counter = 0

            if stagnation_counter > stagnation_patience:
                restart_iter += 1
                xmean = np.random.uniform(self.lb, self.ub, size=self.dim)
                sigma = self.initial_sigma
                C = np.eye(self.dim)
                pc = np.zeros(self.dim)
                ps = np.zeros(self.dim)
                stagnation_counter = 0
                if restart_iter > max_restarts:
                    break

            previous_f_opt = f_opt

            # Update distribution parameters
            xmean_new = np.sum(x[:, :mu_adjusted] * weights[:mu_adjusted], axis=1)

            ps = (1-cs) * ps + np.sqrt(cs*(2-cs)*mueff) * np.dot(np.linalg.inv(np.linalg.cholesky(C)), (xmean_new - xmean)) / sigma
            hsig = np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*evals/lambda_))/chiN < 1.4 + 2/(self.dim+1)
            pc = (1-ccov) * pc + hsig * np.sqrt(ccov*(2-ccov)*mueff) * (xmean_new - xmean) / sigma

            C = (1-c1-cmu) * C + c1 * (np.outer(pc, pc) + (1-hsig) * ccov*(2-ccov) * C)
            C += cmu * np.sum(weights[:mu_adjusted, None, None] * np.array([np.outer(x[:, i] - xmean, x[:, i] - xmean) for i in range(mu_adjusted)]), axis=0) / sigma**2

            sigma = sigma * np.exp((cs/damps) * (np.linalg.norm(ps)/chiN - 1))

            xmean = xmean_new

            # Repair covariance matrix
            C = np.triu(C) + np.transpose(np.triu(C,1))

            # Regularize covariance matrix (Adaptive Regularization)
            C = C + np.eye(self.dim) * 1e-8 * sigma * (1 + evals/self.budget)

            try:
                np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = C + np.eye(self.dim) * 1e-6 * sigma

            # Restart mechanism
            if np.max(np.diag(C)) > (10**7) * sigma:
                restart_iter += 1
                xmean = np.random.uniform(self.lb, self.ub, size=self.dim)
                sigma = self.initial_sigma
                C = np.eye(self.dim)
                pc = np.zeros(self.dim)
                ps = np.zeros(self.dim)
                stagnation_counter = 0
                if restart_iter > max_restarts:
                    break

            if np.any(np.isnan(C)):
                C = np.eye(self.dim)
                sigma = self.initial_sigma
                pc = np.zeros(self.dim)
                ps = np.zeros(self.dim)
                stagnation_counter = 0

        return f_opt, x_opt