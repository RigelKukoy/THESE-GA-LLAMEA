import numpy as np

class CMAES_AdaptiveMirrorRank_ActiveAdaptation_OrthogonalLearning:
    def __init__(self, budget=10000, dim=10, sigma0=0.5, history_length=10, initial_mirrored_fraction=0.5, mirrored_decay=0.99, active_adaptation_multiplier=0.1, orthogonal_learning_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.sigma0 = sigma0
        self.history_length = history_length
        self.mirrored_fraction = initial_mirrored_fraction
        self.mirrored_decay = mirrored_decay
        self.active_adaptation_multiplier = active_adaptation_multiplier
        self.orthogonal_learning_rate = orthogonal_learning_rate

        self.popsize = 4 + int(np.floor(3 * np.log(self.dim)))
        self.mu = self.popsize // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = self.weights / np.sum(self.weights)
        self.mueff = np.sum(self.weights)**2 / np.sum(self.weights**2)
        self.step_history = []
        self.success_rate_history = []
        self.success_rate_window = 10
        self.population_size_adaptation_frequency = 10  # Adjust popsize every this many generations
        self.population_size_scale = 1.0

    def __call__(self, func):
        # Initialize variables
        mean = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
        sigma = self.sigma0
        C = np.eye(self.dim)
        pc = np.zeros(self.dim)
        ps = np.zeros(self.dim)
        chiN = np.sqrt(self.dim) * (1 - (1 / (4 * self.dim)) + 1 / (21 * self.dim**2))

        # Parameters (using common defaults)
        c_sigma = (self.mueff + 2) / (self.dim + self.mueff + 5)
        c_c = (4 + self.mueff / self.dim) / (self.dim + 4 + 2 * self.mueff / self.dim)
        c_1 = 2 / ((self.dim + 1.3)**2 + self.mueff)
        c_mu = min(1 - c_1, 2 * (self.mueff - 1 + 1 / self.mueff) / ((self.dim + 2)**2 + self.mueff))
        d_sigma = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1)
        c_1a = c_1
        c_mua = c_mu

        # Eigen decomposition of C
        C_evals, C_evecs = np.linalg.eigh(C)
        C_sqrt = C_evecs @ np.diag(np.sqrt(C_evals)) @ C_evecs.T
        C_invsqrt = C_evecs @ np.diag(1/np.sqrt(C_evals)) @ C_evecs.T
        
        f_opt = np.Inf
        x_opt = None
        eval_count = 0
        successes = 0
        generation = 0

        while eval_count < self.budget:
            # Adjust population size
            self.popsize = int(self.population_size_scale * (4 + int(np.floor(3 * np.log(self.dim)))))
            self.mu = self.popsize // 2
            self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
            self.weights = self.weights / np.sum(self.weights)
            self.mueff = np.sum(self.weights)**2 / np.sum(self.weights**2)
            
            # Sample population
            z = np.random.normal(0, 1, size=(self.popsize, self.dim))
            x = mean + sigma * (C_sqrt @ z.T).T

            # Mirrored sampling
            num_mirrored = int(self.popsize * self.mirrored_fraction)
            x_mirrored = mean - sigma * (C_sqrt @ z[:num_mirrored].T).T
            x = np.vstack((x, x_mirrored))
            z_mirrored = -z[:num_mirrored]
            z = np.vstack((z, z_mirrored))
            
            # Clipping to bounds
            x = np.clip(x, func.bounds.lb, func.bounds.ub)
            
            # Evaluate the new points
            f = np.array([func(xi) if eval_count + i < self.budget else np.inf for i, xi in enumerate(x)])
            eval_count += len(x) 

            # Sort by fitness
            idx = np.argsort(f)
            x = x[idx]
            z = z[idx]
            f = f[idx]

            # Update optimal solution
            if f[0] < f_opt:
                f_opt = f[0]
                x_opt = x[0]
                successes += 1 # increment successful steps
                
            # Selection and recombination
            x_mu = x[:self.mu]
            z_mu = z[:self.mu]

            mean_new = np.sum(x_mu * self.weights[:,None], axis=0)
            z_w = np.sum(z_mu * self.weights[:,None], axis=0)
            
            # Covariance matrix adaptation using rank-mu update
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * self.mueff) * (C_invsqrt @ (mean_new - mean) / sigma)
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * eval_count / self.popsize)) < chiN * (1.4 + 2 / (self.dim + 1))
            
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * self.mueff) * (mean_new - mean) / sigma

            C = (1 - c_1 - c_mu + self.active_adaptation_multiplier * c_1 * (1 - hsig**2)) * C + c_1 * (pc[:, None] @ pc[None, :])

            # Rank-mu update
            for i in range(self.mu):
                C += c_mu * self.weights[i] * (z_mu[i, :, None] @ z_mu[i, None, :])

            # Orthogonal subspace learning
            if self.orthogonal_learning_rate > 0:
                Q, R = np.linalg.qr(z_mu.T)  # Orthogonal basis
                delta_C = np.zeros_like(C)
                for i in range(Q.shape[1]):
                    delta_C += self.orthogonal_learning_rate * (Q[:, i:i+1] @ Q[:, i:i+1].T)
                C += delta_C
                
            # Active covariance matrix adaptation: encourage/discourage steps
            if f[0] > f_opt:
                 C -= self.active_adaptation_multiplier * c_1 * (pc[:, None] @ pc[None, :])  # Discourage step if fitness worsened
            
            # Update step size: using adaptive success rate
            success_rate = successes / (eval_count / self.popsize)
            self.success_rate_history.append(success_rate)
    
            if len(self.success_rate_history) > self.success_rate_window:
                self.success_rate_history.pop(0)
    
            avg_success_rate = np.mean(self.success_rate_history)
            
            # Adjust sigma based on success rate
            if avg_success_rate > 0.25:
                sigma *= 1.1  # Increase step-size if doing well
            elif avg_success_rate < 0.15:
                sigma *= 0.9  # Decrease step-size if not improving

            # Step history adaptation
            step = (mean_new - mean) / sigma
            self.step_history.append(step)
            if len(self.step_history) > self.history_length:
                self.step_history.pop(0)

            # Adapt C based on step history (dampened)
            for h_step in self.step_history:
                C += 0.1 * c_mu * (h_step[:, None] @ h_step[None, :])  # Dampened update
                
            # Update mean
            mean = mean_new

            # Adaptive Mirrored fraction
            self.mirrored_fraction *= self.mirrored_decay
            
            # Spectral correction to ensure C remains positive definite
            C = np.triu(C) + np.triu(C, 1).T
            C_evals, C_evecs = np.linalg.eigh(C)
            C_evals = np.maximum(C_evals, 1e-10)  # Ensure positive definite
            C = C_evecs @ np.diag(C_evals) @ C_evecs.T
            
            # Eigen decomposition of C
            if eval_count % (self.popsize * 5) == 0:
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    C_evals, C_evecs = np.linalg.eigh(C)
                    C_evals = np.maximum(C_evals, 1e-10)
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
                successes = 0
                self.success_rate_history = []
                
            generation += 1
            if generation % self.population_size_adaptation_frequency == 0:
                if avg_success_rate > 0.3:
                    self.population_size_scale *= 1.1
                elif avg_success_rate < 0.1:
                    self.population_size_scale *= 0.9
                self.population_size_scale = np.clip(self.population_size_scale, 0.5, 2.0)  # Limit population size scaling

        
        return f_opt, x_opt