import numpy as np

class AdaptiveRestartCMAES:
    def __init__(self, budget=10000, dim=10, sigma0 = 0.5, restarts=5):
        self.budget = budget
        self.dim = dim
        self.sigma0 = sigma0
        self.restarts = restarts

    def __call__(self, func):
        f_opt = np.Inf
        x_opt = None
        eval_count = 0
        restart_count = 0
        stagnation_counter = 0
        
        while eval_count < self.budget and restart_count < self.restarts:
            # Adaptive population size
            popsize = 4 + int(np.floor(3 * np.log(self.dim)))
            if stagnation_counter > 5:
                popsize = 4 + int(np.floor(5 * np.log(self.dim))) # Increase popsize if stagnating
            mu = popsize // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights = weights / np.sum(weights)
            mueff = np.sum(weights)**2 / np.sum(weights**2)

            # Initialize variables
            mean = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
            sigma = self.sigma0
            C = np.eye(self.dim)  # Covariance matrix
            pc = np.zeros(self.dim)
            ps = np.zeros(self.dim)
            chiN = np.sqrt(self.dim) * (1 - (1 / (4 * self.dim)) + 1 / (21 * self.dim**2))

            # Parameters (using common defaults)
            c_sigma = (mueff + 2) / (self.dim + mueff + 5)
            c_c = (4 + mueff / self.dim) / (self.dim + 4 + 2 * mueff / self.dim)
            c_1 = 2 / ((self.dim + 1.3)**2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 1 + 1 / mueff) / ((self.dim + 2)**2 + mueff))
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (self.dim + 1)) - 1) + c_sigma
            c_1a = c_1
            c_mua = c_mu

            # Eigen decomposition of C (expensive, do it rarely)
            try:
                C_evals, C_evecs = np.linalg.eigh(C)
                C_sqrt = C_evecs @ np.diag(np.sqrt(C_evals)) @ C_evecs.T
                C_invsqrt = C_evecs @ np.diag(1/np.sqrt(C_evals)) @ C_evecs.T
            except np.linalg.LinAlgError:
                print("LinAlgError encountered during initialization, resetting C")
                C = np.eye(self.dim)
                C_evals, C_evecs = np.linalg.eigh(C)
                C_sqrt = C_evecs @ np.diag(np.sqrt(C_evals)) @ C_evecs.T
                C_invsqrt = C_evecs @ np.diag(1/np.sqrt(C_evals)) @ C_evecs.T
            
            last_f_opt = np.Inf
            local_eval_count = 0
            
            while eval_count < self.budget:
                # Sample population
                z = np.random.normal(0, 1, size=(popsize, self.dim))
                x = mean + sigma * (C_sqrt @ z.T).T
                
                # Evaluate the new points
                f = np.array([func(xi) if eval_count + i < self.budget else np.inf for i, xi in enumerate(x)])
                eval_count += len(x)
                local_eval_count += len(x)

                # Sort by fitness
                idx = np.argsort(f)
                x = x[idx]
                z = z[idx]
                f = f[idx]

                # Update optimal solution
                if f[0] < f_opt:
                    f_opt = f[0]
                    x_opt = x[0]
                    stagnation_counter = 0  # Reset stagnation counter
                else:
                    stagnation_counter += 1

                # Check for stagnation
                if abs(f_opt - last_f_opt) < 1e-9 or stagnation_counter > 20 * popsize: # increased stagnation
                    print(f"Stagnation detected after {local_eval_count} evaluations, restarting CMA-ES.")
                    restart_count += 1
                    break # Restart CMA-ES
                    
                last_f_opt = f_opt
                    
                # Selection and recombination
                x_mu = x[:mu]
                z_mu = z[:mu]

                mean_new = np.sum(x_mu * weights[:,None], axis=0)
                z_w = np.sum(z_mu * weights[:,None], axis=0)
                
                # Covariance matrix adaptation
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_invsqrt @ (mean_new - mean) / sigma)
                hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * eval_count / popsize)) < chiN * (1.4 + 2 / (self.dim + 1))
                
                pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mueff) * (mean_new - mean) / sigma

                C = (1 - c_1 - c_mu) * C + c_1 * (pc[:, None] @ pc[None, :])
                for i in range(mu):
                    C += c_mu * weights[i] * (z_mu[i, :, None] @ z_mu[i, None, :])
                    
                # Active CMA
                if c_1a > 0 and c_mua > 0:
                    negidx = np.where(weights < 0)[0]
                    znorm = np.zeros((len(negidx), self.dim))
                    for i, idx in enumerate(negidx):
                        znorm[i] = z_mu[idx] * np.sqrt(-weights[idx])
                    C += c_1a * (1/np.linalg.norm(ps)**2 + c_mua) * np.sum(znorm.T @ znorm, axis=1)

                # Update step size
                sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(ps) / chiN - 1))
                    
                # Update mean
                mean = mean_new

                # Eigen decomposition of C
                if local_eval_count % (popsize * 5) == 0:  # Re-compute after every 5 generations
                    C = np.triu(C) + np.triu(C, 1).T  # Enforce symmetry
                    try:
                        C_evals, C_evecs = np.linalg.eigh(C)
                        C_evals = np.maximum(C_evals, 1e-10) # Avoid zero or negative eigenvalues
                        C = C_evecs @ np.diag(C_evals) @ C_evecs.T
                        C_sqrt = C_evecs @ np.diag(np.sqrt(C_evals)) @ C_evecs.T
                        C_invsqrt = C_evecs @ np.diag(1/np.sqrt(C_evals)) @ C_evecs.T
                    except np.linalg.LinAlgError:
                        print("LinAlgError encountered, resetting C")
                        C = np.eye(self.dim)
                        C_evals, C_evecs = np.linalg.eigh(C)
                        C_sqrt = C_evecs @ np.diag(np.sqrt(C_evals)) @ C_evecs.T
                        C_invsqrt = C_evecs @ np.diag(1/np.sqrt(C_evals)) @ C_evecs.T
                    
                if np.any(np.isnan(mean)) or np.any(np.isnan(C)):
                    print("NaN detected, resetting...")
                    mean = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
                    C = np.eye(self.dim)
                    pc = np.zeros(self.dim)
                    ps = np.zeros(self.dim)
                    C_evals, C_evecs = np.linalg.eigh(C)
                    C_sqrt = C_evecs @ np.diag(np.sqrt(C_evals)) @ C_evecs.T
                    C_invsqrt = C_evecs @ np.diag(1/np.sqrt(C_evals)) @ C_evecs.T
                    sigma = self.sigma0
                    stagnation_counter = 0
                    
        return f_opt, x_opt