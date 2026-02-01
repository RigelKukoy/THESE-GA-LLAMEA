import numpy as np

class CMAES:
    def __init__(self, budget=10000, dim=10, mu_ratio=0.25, cs=0.3, dsigma=0.2, ccov=0.3, initial_sigma=0.5, target_success_rate=0.25, c_alpha=0.1):
        self.budget = budget
        self.dim = dim
        self.mu_ratio = mu_ratio
        self.cs = cs
        self.dsigma = dsigma
        self.ccov = ccov
        self.lb = -5.0
        self.ub = 5.0
        self.initial_sigma = initial_sigma
        self.target_success_rate = target_success_rate
        self.c_alpha = c_alpha # Learning rate for active CMA

    def __call__(self, func):
        # Initialize variables
        xmean = np.random.uniform(self.lb, self.ub, size=self.dim)  # Initial guess of mean
        sigma = self.initial_sigma  # Overall step size

        lambda_ = int(4 + np.floor(3 * np.log(self.dim)))  # Initial population size
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

        # Active CMA parameters
        c_alpha = self.c_alpha
        mu_neg = int(lambda_ / 2)
        weights_neg = -weights[:mu_neg]
        mueff_neg = np.sum(weights_neg)**2 / np.sum(weights_neg**2)

        f_opt = np.Inf
        x_opt = None
        evals = 0

        restart_iter = 0
        max_restarts = 5
        no_improvement_counter = 0
        no_improvement_threshold = 1000 #Function evaluations without improvement
        success_rate = self.target_success_rate

        adapt_lambda_iter = 0
        adapt_lambda_freq = 500

        while evals < self.budget:
            # Generate and evaluate offspring (mirrored sampling)
            z = np.random.normal(0, 1, size=(self.dim, lambda_))
            y = np.dot(np.linalg.cholesky(C), z)
            x = xmean.reshape(-1, 1) + sigma * y
            x_mirrored = xmean.reshape(-1, 1) - sigma * y
            x = np.concatenate((x, x_mirrored), axis=1)
            x = np.clip(x, self.lb, self.ub)

            f = np.array([func(x[:,i]) if evals + i < self.budget else np.inf for i in range(x.shape[1])])
            evals += x.shape[1]
            
            # Sort by fitness
            idx = np.argsort(f)
            f = f[idx]
            x = x[:, idx]

            # Update optimal solution
            if f[0] < f_opt:
                f_opt = f[0]
                x_opt = x[:, 0].copy()  # Ensure x_opt is a copy
                no_improvement_counter = 0
            else:
                no_improvement_counter += x.shape[1]

            # Update distribution parameters
            xmean_new = np.sum(x[:, :mu] * weights, axis=1)

            ps = (1-cs) * ps + np.sqrt(cs*(2-cs)*mueff) * np.dot(np.linalg.inv(np.linalg.cholesky(C)), (xmean_new - xmean)) / sigma
            hsig = np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*evals/lambda_))/chiN < 1.4 + 2/(self.dim+1)
            pc = (1-ccov) * pc + hsig * np.sqrt(ccov*(2-ccov)*mueff) * (xmean_new - xmean) / sigma

            C = (1-c1-cmu) * C + c1 * (np.outer(pc, pc) + (1-hsig) * ccov*(2-ccov) * C)

            # Active CMA
            y_neg = (x[:, -mu_neg:] - xmean.reshape(-1, 1)) / sigma
            C += cmu * np.sum(weights[:, None, None] * np.array([np.outer(x[:, i] - xmean, x[:, i] - xmean) for i in range(mu)]), axis=0) / sigma**2
            C += c_alpha * (1 - c1 - cmu) * np.sum(weights_neg[:, None, None] * np.array([np.outer(y_neg[:, i], y_neg[:, i]) for i in range(mu_neg)]), axis=0)

            # Spectral correction of C
            try:
                d, B = np.linalg.eigh(C)
                d = np.real(d)
                B = np.real(B)
                d[d < 0] = 1e-8  # prevent negative eigenvalues
                C = B.dot(np.diag(d)).dot(B.T)
            except np.linalg.LinAlgError:
                C = C + np.eye(self.dim) * 1e-6

            sigma = sigma * np.exp((cs/damps) * (np.linalg.norm(ps)/chiN - 1))

            # Adapt step size based on success rate
            success_rate = 0.9 * success_rate + 0.1 * (f[0] < f_opt) # smoothed success rate

            if success_rate > self.target_success_rate:
                sigma *= np.exp(0.1 * (success_rate - self.target_success_rate) / self.target_success_rate)
            else:
                sigma *= np.exp(0.1 * (success_rate - self.target_success_rate) / (1 - self.target_success_rate))

            xmean = xmean_new

            # Repair covariance matrix (ensure positive definiteness)
            C = np.triu(C) + np.transpose(np.triu(C,1))
            try:
                np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = C + np.eye(self.dim) * 1e-6

            # Adapt population size
            adapt_lambda_iter += x.shape[1]
            if adapt_lambda_iter > adapt_lambda_freq:
                adapt_lambda_iter = 0
                if success_rate > self.target_success_rate:
                    lambda_ = int(lambda_ * 1.1)
                else:
                    lambda_ = int(lambda_ * 0.9)
                lambda_ = max(4, lambda_)
                mu = int(lambda_ * self.mu_ratio)
                weights = np.log(mu + 1/2) - np.log(np.arange(1, mu + 1))  # Weights for recombination
                weights = weights / np.sum(weights)
                mueff = np.sum(weights)**2 / np.sum(weights**2)
                mu_neg = int(lambda_ / 2)
                weights_neg = -weights[:mu_neg]
                mueff_neg = np.sum(weights_neg)**2 / np.sum(weights_neg**2)


            # Restart mechanism
            if np.max(np.diag(C)) > (10**7) * sigma or sigma < 1e-10 or no_improvement_counter > no_improvement_threshold:
                restart_iter += 1
                xmean = np.random.uniform(self.lb, self.ub, size=self.dim)
                sigma = self.initial_sigma
                C = np.eye(self.dim)
                pc = np.zeros(self.dim)
                ps = np.zeros(self.dim)
                no_improvement_counter = 0 # Reset no improvement counter
                if restart_iter > max_restarts:
                    break

            if np.any(np.isnan(C)):
                C = np.eye(self.dim)
                sigma = self.initial_sigma
                pc = np.zeros(self.dim)
                ps = np.zeros(self.dim)
                no_improvement_counter = 0
        return f_opt, x_opt