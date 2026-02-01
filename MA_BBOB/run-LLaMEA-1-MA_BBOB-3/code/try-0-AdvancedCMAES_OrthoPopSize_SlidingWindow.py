import numpy as np

class AdvancedCMAES_OrthoPopSize_SlidingWindow(object):
    def __init__(self, budget=10000, dim=10, sigma0=0.5, history_length=5, mirror_ratio=0.5, restart_strategy="random", subspace_size=None):
        self.budget = budget
        self.dim = dim
        self.sigma0 = sigma0
        self.history_length = history_length
        self.min_popsize = 4
        self.max_popsize = 50
        self.target_popsize = min(self.max_popsize, self.min_popsize + int(np.floor(3 * np.log(self.dim))))
        self.popsize = self.target_popsize
        self.mu = self.popsize // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = self.weights / np.sum(self.weights)
        self.mueff = np.sum(self.weights)**2 / np.sum(self.weights**2)
        self.step_history = []
        self.last_f_opt = np.inf
        self.stagnation_counter = 0
        self.restart_iterations = 10 * self.dim  # More aggressive stagnation detection
        self.mirror_ratio = mirror_ratio
        self.restart_strategy = restart_strategy # "random" or "jitter"
        self.condition_number_target = 1e14
        self.damps = 1.0 # Damping factor for step size
        self.cs = 0.3  # Step-size learning rate
        self.cc = 0.4  # Cumulation for covariance matrix rank-one update
        self.c1 = 1.5 / (self.dim + 2)**2  # Learning rate for rank-one update
        self.cmu = 1.5 * self.mueff / (self.dim + 2)**2  # Learning rate for rank-mu update
        self.init_lambda = 2 + np.floor(3 * np.log(self.dim))
        self.lambda_factor = 2
        self.orthogonal_basis = np.random.randn(self.dim, self.dim)
        self.orthogonal_basis, _ = np.linalg.qr(self.orthogonal_basis)
        self.mean_history = []
        self.subspace_size = subspace_size if subspace_size is not None else min(self.dim // 2, 10)

    def __call__(self, func):
        mean = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
        sigma = self.sigma0
        C = np.eye(self.dim)
        pc = np.zeros(self.dim)
        ps = np.zeros(self.dim)
        chiN = np.sqrt(self.dim) * (1 - (1 / (4 * self.dim)) + 1 / (21 * self.dim**2))

        C_evals, C_evecs = np.linalg.eigh(C)
        C_sqrt = C_evecs @ np.diag(np.sqrt(C_evals)) @ C_evecs.T
        C_invsqrt = C_evecs @ np.diag(1 / np.sqrt(C_evals)) @ C_evecs.T
        
        f_opt = np.Inf
        x_opt = None
        eval_count = 0

        while eval_count < self.budget:
            # Adapt population size
            self.target_popsize = min(self.max_popsize, self.min_popsize + int(np.floor(3 * np.log(self.dim))))
            if self.target_popsize != self.popsize:
                self.popsize = self.target_popsize
                self.mu = self.popsize // 2
                self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
                self.weights = self.weights / np.sum(self.weights)
                self.mueff = np.sum(self.weights)**2 / np.sum(self.weights**2)
                self.cmu = 1.5 * self.mueff / (self.dim + 2)**2

            z = np.random.normal(0, 1, size=(self.popsize, self.dim))
            x = mean + sigma * (C_sqrt @ z.T).T

            # Mirrored Sampling
            num_mirrored = int(self.popsize * self.mirror_ratio)
            z_mirrored = -z[:num_mirrored]
            x_mirrored = mean + sigma * (C_sqrt @ z_mirrored.T).T
            x = np.concatenate((x, x_mirrored), axis=0)
            z = np.concatenate((z, z_mirrored), axis=0)

            f = np.array([func(xi) if eval_count + i < self.budget else np.inf for i, xi in enumerate(x)])
            eval_count += len(x)

            idx = np.argsort(f)
            x = x[idx]
            z = z[idx]
            f = f[idx]

            if f[0] < f_opt:
                f_opt = f[0]
                x_opt = x[0]
                self.stagnation_counter = 0
                self.last_f_opt = f[0] #Store for stagnation criterion
            else:
                self.stagnation_counter += 1
            
            # Restart strategy
            if self.stagnation_counter > self.restart_iterations:
                if self.restart_strategy == "random":
                    mean = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
                elif self.restart_strategy == "jitter":
                     mean = mean + 0.1 * np.random.normal(0, 1, size=self.dim) #Jitter the mean
                     mean = np.clip(mean, func.bounds.lb, func.bounds.ub) #Ensure bounds
                else:
                    raise ValueError("Invalid restart strategy.")

                sigma = self.sigma0
                C = np.eye(self.dim)
                pc = np.zeros(self.dim)
                ps = np.zeros(self.dim)
                C_evals, C_evecs = np.linalg.eigh(C)
                C_sqrt = C_evecs @ np.diag(np.sqrt(C_evals)) @ C_evecs.T
                C_invsqrt = C_evecs @ np.diag(1 / np.sqrt(C_evals)) @ C_evecs.T
                self.stagnation_counter = 0

            x_mu = x[:self.mu]
            z_mu = z[:self.mu]

            mean_new = np.sum(x_mu * self.weights[:, None], axis=0)
            z_w = np.sum(z_mu * self.weights[:, None], axis=0)
            
            ps = (1 - self.cs) * ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * (C_invsqrt @ (mean_new - mean) / sigma)
            
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - self.cs)**(2 * eval_count / self.popsize)) < chiN * (1.4 + 2 / (self.dim + 1))
            
            pc = (1 - self.cc) * pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * (mean_new - mean) / sigma

            # Covariance matrix update with spectral decay
            C = (1 - self.c1 - self.cmu) * C + self.c1 * (pc[:, None] @ pc[None, :])
            for i in range(self.mu):
                C += self.cmu * self.weights[i] * (z_mu[i, :, None] @ z_mu[i, None, :])

            # Ensure symmetry
            C = np.triu(C) + np.triu(C, 1).T
            
            sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma *= self.damps # Apply damping

            # Store mean and update orthogonal subspace
            self.mean_history.append(mean_new)
            if len(self.mean_history) > self.history_length:
                self.mean_history.pop(0)

            # Orthogonal subspace learning every 10 generations
            if eval_count % (10 * self.popsize) == 0 and len(self.mean_history) > 1:
                # Construct a matrix of recent search steps
                search_directions = np.array(self.mean_history[1:]) - np.array(self.mean_history[:-1])
                
                # Perform SVD to find the dominant directions
                try:
                    U, S, V = np.linalg.svd(search_directions.T)
                    dominant_directions = U[:, :self.subspace_size]  # Select top subspace_size directions

                    # Update orthogonal basis with the dominant directions
                    self.orthogonal_basis[:, :self.subspace_size] = dominant_directions

                except np.linalg.LinAlgError:
                    pass # Handle potential SVD issues

                # Project and update (same as before)
                subspace_size = self.subspace_size
                Q = self.orthogonal_basis[:, :subspace_size]
                y = Q.T @ (mean_new - mean) #Project into subspace
                C_subspace = Q.T @ C @ Q #Project covariance matrix
                try:
                    C_evals_sub, C_evecs_sub = np.linalg.eigh(C_subspace) #Eigen-decomposition
                    Q_update = Q @ C_evecs_sub #Update vectors
                    self.orthogonal_basis[:, :subspace_size] = Q_update #Update orthogonal basis
                except np.linalg.LinAlgError:
                    pass #Do not update

            # Regularize and update covariance matrix
            if eval_count % (self.popsize * 5) == 0:
                C = np.triu(C) + np.triu(C, 1).T
                try:
                    C_evals, C_evecs = np.linalg.eigh(C)
                    C_evals = np.maximum(C_evals, 1e-10)

                    # Spectral Decay: Shrink smaller eigenvalues
                    C_evals = C_evals * (1 - 0.01 * np.exp(-np.arange(self.dim)))

                    C = C_evecs @ np.diag(C_evals) @ C_evecs.T
                    C_sqrt = C_evecs @ np.diag(np.sqrt(C_evals)) @ C_evecs.T
                    C_invsqrt = C_evecs @ np.diag(1 / np.sqrt(C_evals)) @ C_evecs.T

                     # Condition number based regularization
                    condition_number = np.max(C_evals) / np.min(C_evals)
                    if condition_number > self.condition_number_target:
                        regularization_factor = 1e-8 * (condition_number / self.condition_number_target)
                        C = C + regularization_factor * np.eye(self.dim)

                except np.linalg.LinAlgError:
                    C = np.eye(self.dim)
                    C_evals, C_evecs = np.linalg.eigh(C)
                    C_sqrt = C_evecs @ np.diag(np.sqrt(C_evals)) @ C_evecs.T
                    C_invsqrt = C_evecs @ np.diag(1 / np.sqrt(C_evals)) @ C_evecs.T

                # Regularize Covariance Matrix
                C = C + 1e-8 * np.eye(self.dim)
                
            if np.any(np.isnan(mean)) or np.any(np.isnan(C)):
                mean = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
                C = np.eye(self.dim)
                pc = np.zeros(self.dim)
                ps = np.zeros(self.dim)
                C_evals, C_evecs = np.linalg.eigh(C)
                C_sqrt = C_evecs @ np.diag(np.sqrt(C_evals)) @ C_evecs.T
                C_invsqrt = C_evecs @ np.diag(1 / np.sqrt(C_evals)) @ C_evecs.T
                sigma = self.sigma0

            mean = mean_new

        return f_opt, x_opt