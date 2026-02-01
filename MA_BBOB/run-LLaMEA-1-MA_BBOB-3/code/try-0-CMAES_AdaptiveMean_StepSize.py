import numpy as np

class CMAES_AdaptiveMean_StepSize:
    def __init__(self, budget=10000, dim=10, sigma0 = 0.5):
        self.budget = budget
        self.dim = dim
        self.sigma0 = sigma0

        self.popsize = 4 + int(np.floor(3 * np.log(self.dim)))
        self.mu = self.popsize // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = self.weights / np.sum(self.weights)
        self.mueff = np.sum(self.weights)**2 / np.sum(self.weights**2)

        self.success_rate = 0.2  # Initial success rate
        self.learning_rate_sigma = 0.2  # Learning rate for step size

        self.C_update_frequency = self.popsize * 5  # Initial frequency
        self.C_update_factor = 2 # Factor to increase/decrease frequency

        self.f_opt_history = []
        self.history_length = 10
        self.mean_history = []
        
        self.population_size_history = []
        self.adapt_popsize = True
        self.min_popsize = 4 + int(np.floor(3 * np.log(self.dim)))
        self.max_popsize = 4 + int(np.floor(10 * np.log(self.dim)))

    def __call__(self, func):
        # Initialize variables
        mean = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
        sigma = self.sigma0
        C = np.eye(self.dim)  # Covariance matrix
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

        # Eigen decomposition of C (expensive, do it rarely)
        try:
            C_evals, C_evecs = np.linalg.eigh(C)
            C_evals = np.maximum(C_evals, 1e-10)
            C_sqrt = C_evecs @ np.diag(np.sqrt(C_evals)) @ C_evecs.T
            C_invsqrt = C_evecs @ np.diag(1/np.sqrt(C_evals)) @ C_evecs.T
        except np.linalg.LinAlgError:
            print("Initial LinAlgError encountered, resetting C")
            C = np.eye(self.dim)
            C_evals, C_evecs = np.linalg.eigh(C)
            C_sqrt = C_evecs @ np.diag(np.sqrt(C_evals)) @ C_evecs.T
            C_invsqrt = C_evecs @ np.diag(1/np.sqrt(C_evals)) @ C_evecs.T

        f_opt = np.Inf
        x_opt = None
        eval_count = 0
        successes = 0
        step_size_adaptation_history = [] # Store past step sizes.

        while eval_count < self.budget:
            # Sample population
            z = np.random.normal(0, 1, size=(self.popsize, self.dim))
            x = mean + sigma * (C_sqrt @ z.T).T
            
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
                successes += 1
                
            # Selection and recombination
            x_mu = x[:self.mu]
            z_mu = z[:self.mu]

            mean_new = np.sum(x_mu * self.weights[:,None], axis=0)
            z_w = np.sum(z_mu * self.weights[:,None], axis=0)
            
            # Covariance matrix adaptation
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * self.mueff) * (C_invsqrt @ (mean_new - mean) / sigma)
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * eval_count / self.popsize)) < chiN * (1.4 + 2 / (self.dim + 1))
            
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * self.mueff) * (mean_new - mean) / sigma

            C = (1 - c_1 - c_mu) * C + c_1 * (pc[:, None] @ pc[None, :])
            for i in range(self.mu):
                C += c_mu * self.weights[i] * (z_mu[i, :, None] @ z_mu[i, None, :])
                
            # Active CMA
            if c_1a > 0 and c_mua > 0:
                negidx = np.where(self.weights < 0)[0]
                znorm = np.zeros((len(negidx), self.dim))
                for i, idx in enumerate(negidx):
                    znorm[i] = z_mu[idx] * np.sqrt(-self.weights[idx])
                C += c_1a * (1/np.linalg.norm(ps)**2 + c_mua) * np.sum(znorm.T @ znorm, axis=1)
            
            # Spectral Clipping (Correct negative eigenvalues)
            C = np.triu(C) + np.triu(C, 1).T  # Enforce symmetry
            try:
                evals, evecs = np.linalg.eigh(C)
                evals = np.maximum(evals, 1e-10)  # Clip small/negative eigenvalues
                C = evecs @ np.diag(evals) @ evecs.T
            except np.linalg.LinAlgError:
                print("LinAlgError during spectral clipping, resetting C")
                C = np.eye(self.dim)

            # Update success rate
            self.success_rate = 0.9 * self.success_rate + 0.1 * (successes > 0) # Binary success or failure
            successes = 0

            # Population based step size adaptation
            delta_f = np.mean(f) - f_opt # Average fitness of the population minus best fitness
            if len(step_size_adaptation_history) > 5:
                # Use the average of the step sizes
                avg_sigma = np.mean(step_size_adaptation_history[-5:])
                if delta_f > 0:
                    sigma = avg_sigma * np.exp(self.learning_rate_sigma * (self.success_rate - 0.2)) # Reduce
                else:
                    sigma = avg_sigma * np.exp(self.learning_rate_sigma * (self.success_rate + 0.2)) # Increase
            else:
                sigma *= np.exp(self.learning_rate_sigma * (self.success_rate - 0.2))  # Adjust towards target rate of 0.2
            step_size_adaptation_history.append(sigma)
            
            # Dynamic Mean Adaptation: If improvement stagnates, perturb the mean.
            self.f_opt_history.append(f_opt)
            self.mean_history.append(mean)
            if len(self.f_opt_history) > self.history_length:
                self.f_opt_history.pop(0)
                self.mean_history.pop(0)
                
                change = abs(self.f_opt_history[-1] - self.f_opt_history[0])

                if change < 1e-5: # Very small improvement, consider perturbing.
                    # Perturb mean, with decaying perturbation size
                    perturbation = np.random.normal(0, sigma * 0.1, size=self.dim) # Scale with current sigma
                    mean = self.mean_history[-1] + perturbation # Perturb the mean
                    
                    # Clip to bounds
                    mean = np.clip(mean, func.bounds.lb, func.bounds.ub)

            # Update mean
            mean = mean_new

            # Adaptive C update frequency
            if len(self.f_opt_history) > self.history_length:
                # Calculate the change in f_opt
                change = abs(self.f_opt_history[-1] - self.f_opt_history[0])

                # Adjust the update frequency based on the change.
                if change < 1e-3: # Stagnation: Reduce frequency.
                    self.C_update_frequency = int(min(self.C_update_frequency * self.C_update_factor, self.budget))
                else: # Improvement: Increase frequency
                    self.C_update_frequency = int(max(self.C_update_frequency / self.C_update_factor, self.popsize))
                    
            # Population size adaptation
            if self.adapt_popsize:
                self.population_size_history.append(f_opt)
                if len(self.population_size_history) > self.history_length:
                    self.population_size_history.pop(0)
                    
                    improvement = abs(self.population_size_history[-1] - self.population_size_history[0])

                    if improvement < 1e-5: # Stagnation
                        self.popsize = max(self.min_popsize, self.popsize // 2) # Reduce population size
                        self.mu = self.popsize // 2
                        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
                        self.weights = self.weights / np.sum(self.weights)
                        self.mueff = np.sum(self.weights)**2 / np.sum(self.weights**2)
                    else:
                        self.popsize = min(self.max_popsize, self.popsize * 2) # Increase population size
                        self.mu = self.popsize // 2
                        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
                        self.weights = self.weights / np.sum(self.weights)
                        self.mueff = np.sum(self.weights)**2 / np.sum(self.weights**2)
                        
            # Eigen decomposition of C
            if eval_count % self.C_update_frequency == 0:  # Re-compute after a dynamic number of generations
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