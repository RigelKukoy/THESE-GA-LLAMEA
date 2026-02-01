import numpy as np

class CMAES_DynamicPopSizeMirrorSigmoid_SelfAdaptC1Cmu:
    def __init__(self, budget=10000, dim=10, sigma0=0.5, history_length=10, initial_mirrored_fraction=0.5, mirrored_decay=0.99, initial_c1=None, initial_cmu=None, popsize_multiplier=1.0):
        self.budget = budget
        self.dim = dim
        self.sigma0 = sigma0
        self.history_length = history_length
        self.mirrored_fraction = initial_mirrored_fraction  # Initial fraction of population to mirror
        self.mirrored_decay = mirrored_decay  # Decay rate for mirrored fraction
        self.popsize_multiplier = popsize_multiplier # Multiplier for adjusting popsize

        self.base_popsize = 4 + int(np.floor(3 * np.log(self.dim)))
        self.popsize = int(self.base_popsize * self.popsize_multiplier)
        self.mu = self.popsize // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = self.weights / np.sum(self.weights)
        self.mueff = np.sum(self.weights)**2 / np.sum(self.weights**2)
        self.step_history = []

        # Initialize c_1 and c_mu
        if initial_c1 is None:
            self.c_1 = 2 / ((self.dim + 1.3)**2 + self.mueff)
        else:
            self.c_1 = initial_c1
        if initial_cmu is None:
            self.c_mu = min(1 - self.c_1, 2 * (self.mueff - 1 + 1 / self.mueff) / ((self.dim + 2)**2 + self.mueff))
        else:
            self.c_mu = initial_cmu

        self.damp_C = 0.2 + self.c_1 + self.c_mu  # Damping factor for covariance matrix update
        self.c_sigma = (self.mueff + 2) / (self.dim + self.mueff + 5)
        self.c_c = (4 + self.mueff / self.dim) / (self.dim + 4 + 2 * self.mueff / self.dim)
        self.d_sigma = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1)

    def __call__(self, func):
        # Initialize variables
        mean = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
        sigma = self.sigma0
        C = np.eye(self.dim)
        pc = np.zeros(self.dim)
        ps = np.zeros(self.dim)
        chiN = np.sqrt(self.dim) * (1 - (1 / (4 * self.dim)) + 1 / (21 * self.dim**2))

        # Eigen decomposition of C
        C_evals, C_evecs = np.linalg.eigh(C)
        C_sqrt = C_evecs @ np.diag(np.sqrt(C_evals)) @ C_evecs.T
        C_invsqrt = C_evecs @ np.diag(1/np.sqrt(C_evals)) @ C_evecs.T
        
        f_opt = np.Inf
        x_opt = None
        eval_count = 0
        stagnation_counter = 0
        previous_f_opt = np.Inf

        while eval_count < self.budget:
            # Sample population
            z = np.random.normal(0, 1, size=(self.popsize, self.dim))
            x = mean + sigma * (C_sqrt @ z.T).T

            # Mirrored sampling with sigmoid adaptation
            mirrored_fraction = self.mirrored_fraction / (1 + np.exp(5 - 10*(eval_count / self.budget)))  # Sigmoid decay
            num_mirrored = int(self.popsize * mirrored_fraction)
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
                stagnation_counter = 0 # Reset stagnation counter
            else:
                stagnation_counter += 1 # increment stagnation counter

            # Adjust population size based on stagnation
            if stagnation_counter > 5 * self.dim:
                self.popsize_multiplier = max(0.5, self.popsize_multiplier * 0.9)
                stagnation_counter = 0
            elif f_opt < previous_f_opt:
                 self.popsize_multiplier = min(2.0, self.popsize_multiplier * 1.1)
            
            self.popsize = int(self.base_popsize * self.popsize_multiplier)
            self.mu = self.popsize // 2
            self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
            self.weights = self.weights / np.sum(self.weights)
            self.mueff = np.sum(self.weights)**2 / np.sum(self.weights**2)

            previous_f_opt = f_opt

            # Selection and recombination
            x_mu = x[:self.mu]
            z_mu = z[:self.mu]

            mean_new = np.sum(x_mu * self.weights[:,None], axis=0)
            z_w = np.sum(z_mu * self.weights[:,None], axis=0)
            
            # Covariance matrix adaptation using rank-mu update
            ps = (1 - self.c_sigma) * ps + np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mueff) * (C_invsqrt @ (mean_new - mean) / sigma)
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - self.c_sigma)**(2 * eval_count / self.popsize)) < chiN * (1.4 + 2 / (self.dim + 1))
            
            pc = (1 - self.c_c) * pc + hsig * np.sqrt(self.c_c * (2 - self.c_c) * self.mueff) * (mean_new - mean) / sigma

            C = (1 - self.c_1 - self.c_mu + self.damp_C * self.c_1) * C + self.c_1 * (pc[:, None] @ pc[None, :])

            # Rank-mu update
            for i in range(self.mu):
                C += self.c_mu * self.weights[i] * (z_mu[i, :, None] @ z_mu[i, None, :])
            
            # Update step size using conjugate evolution path
            step = (mean_new - mean) / sigma
            self.step_history.append(step)
            if len(self.step_history) > self.history_length:
                self.step_history.pop(0)

            # Adapt step size based on step history
            step_correlation = 0
            for h_step in self.step_history:
                step_correlation += np.dot(step, h_step)
            
            sigma_factor = np.exp(0.1 * step_correlation / self.dim)
            sigma *= np.exp((self.c_sigma / self.d_sigma) * (np.linalg.norm(ps) / chiN - 1)) * sigma_factor

            # Update mean
            mean = mean_new

            # Adaptive Mirrored fraction (decay moved to start)
            self.mirrored_fraction *= self.mirrored_decay

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

                # Self-adaptation of c_1 and c_mu based on eigenvalue distribution of C
                eigenvalues = np.linalg.eigvalsh(C)
                condition_number = np.max(eigenvalues) / np.min(eigenvalues)
                learning_rate = 0.1  # Adjust as needed
                self.c_1 *= np.exp(learning_rate * (condition_number - 1))  # Reduce c_1 when C is ill-conditioned
                self.c_mu *= np.exp(-learning_rate * (condition_number - 1)) # Increase c_mu to trust population

                # Ensure c_1 and c_mu remain within reasonable bounds
                self.c_1 = np.clip(self.c_1, 1e-8, 1.0)
                self.c_mu = np.clip(self.c_mu, 1e-8, 1.0)

                self.damp_C = 0.2 + self.c_1 + self.c_mu  # Update damping factor

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
        
        return f_opt, x_opt