import numpy as np

class CMAES_OLS_Mirrored_RankWeighted_Archive:
    def __init__(self, budget=10000, dim=10, mu_ratio=0.25, cs=0.3, dsigma=0.2, ccov=0.3, initial_sigma=0.5, archive_size=100):
        self.budget = budget
        self.dim = dim
        self.mu_ratio = mu_ratio
        self.cs = cs
        self.dsigma = dsigma
        self.ccov = ccov
        self.lb = -5.0
        self.ub = 5.0
        self.initial_sigma = initial_sigma
        self.archive_size = archive_size
        self.archive_x = []
        self.archive_f = []

        self.lambda_ = int(4 + np.floor(3 * np.log(self.dim)))  # Initial population size
        self.mu = int(self.lambda_ * self.mu_ratio)  # Number of parents
        self.weights = np.log(self.mu + 1/2) - np.log(np.arange(1, self.mu + 1))  # Weights for recombination
        self.weights = self.weights / np.sum(self.weights)
        self.mueff = np.sum(self.weights)**2 / np.sum(self.weights**2)
        self.chiN = np.sqrt(self.dim) * (1 - (1/(4*self.dim)) + 1/(12*self.dim**2))
        self.cs_damps = 1 + self.dsigma * max(0, np.sqrt((self.mueff-1)/(self.dim+1))-1) + self.cs  # Damping for step-size
        self.c1 = self.ccov / ((self.dim+1.3)**2 + self.mueff)
        self.cmu = min(1-self.c1, self.ccov * (self.mueff-2+1/self.mueff) / ((self.dim+2.0)**2 + self.mueff))
        self.stagnation_threshold = 50
        self.orthogonal_learning_threshold = 100
        self.max_restarts = 5
        self.mirror_prob = 0.2  # Probability of mirrored sampling
        self.step_size_adaptation_threshold = 10
        self.rank_weight_exponent = 2 #Exponent for rank-based weights
        self.diversity_threshold = 0.1  # Minimum distance for considering a solution diverse

    def __call__(self, func):
        # Initialize variables
        xmean = np.random.uniform(self.lb, self.ub, size=self.dim)  # Initial guess of mean
        sigma = self.initial_sigma  # Overall step size
        C = np.eye(self.dim)  # Covariance matrix
        pc = np.zeros(self.dim)  # Evolution path for C
        ps = np.zeros(self.dim)  # Evolution path for sigma
        
        f_opt = np.Inf
        x_opt = None
        evals = 0

        restart_iter = 0
        stagnation_counter = 0
        orthogonal_learning_counter = 0
        performance_history = []  # Track last 'n' best fitness values
        step_size_history = []

        while evals < self.budget:
            # Generate and evaluate offspring
            z = np.random.normal(0, 1, size=(self.dim, self.lambda_))
            y = np.dot(np.linalg.cholesky(C), z)
            x = xmean.reshape(-1, 1) + sigma * y

            # Mirrored Sampling
            x_mirrored = xmean.reshape(-1, 1) - sigma * y
            
            # Clip and evaluate
            x = np.clip(x, self.lb, self.ub)
            x_mirrored = np.clip(x_mirrored, self.lb, self.ub)
            
            f = np.array([func(x[:,i]) if evals + i < self.budget else np.inf for i in range(self.lambda_//2)])
            f_mirrored = np.array([func(x_mirrored[:,i]) if evals + self.lambda_//2 + i < self.budget else np.inf for i in range(self.lambda_//2)])

            evals += self.lambda_
            f = np.concatenate([f, f_mirrored])
            x = np.concatenate([x[:,:self.lambda_//2], x_mirrored[:,:self.lambda_//2]], axis=1)
            
            # Sort by fitness
            idx = np.argsort(f)
            f = f[idx]
            x = x[:, idx]
            
            # Update optimal solution
            if f[0] < f_opt:
                f_opt = f[0]
                x_opt = x[:, 0].copy()  # Ensure x_opt is a copy
                stagnation_counter = 0  # Reset stagnation counter
                performance_history.append(f_opt)
                step_size_history.append(sigma)

                # Adaptive Lambda: Increase lambda if consistently improving
                if len(performance_history) > 5 and all(performance_history[i] > performance_history[i+1] for i in range(len(performance_history)-1)):
                    self.lambda_ = int(self.lambda_ * 1.1)  # Increase population size by 10%
                    self.mu = int(self.lambda_ * self.mu_ratio)  # Update mu
                    self.weights = np.log(self.mu + 1/2) - np.log(np.arange(1, self.mu + 1))  # Weights for recombination
                    self.weights = self.weights / np.sum(self.weights)
                    self.mueff = np.sum(self.weights)**2 / np.sum(self.weights**2)
                    self.c1 = self.ccov / ((self.dim+1.3)**2 + self.mueff)
                    self.cmu = min(1-self.c1, self.ccov * (self.mueff-2+1/self.mueff) / ((self.dim+2.0)**2 + self.mueff))

            else:
                stagnation_counter += 1
                if len(performance_history) > 0:
                    performance_history.pop(0)  # Remove oldest entry
                if len(step_size_history) > 0:
                    step_size_history.pop(0)

            # Rank-based weighting for covariance matrix adaptation
            rank_weights = np.arange(self.mu, 0, -1) ** self.rank_weight_exponent  # Rank weights
            rank_weights = rank_weights / np.sum(rank_weights)  # Normalize

            # Update distribution parameters
            xmean_new = np.sum(x[:, :self.mu] * self.weights, axis=1)
            
            ps = (1-self.cs) * ps + np.sqrt(self.cs*(2-self.cs)*self.mueff) * np.dot(np.linalg.inv(np.linalg.cholesky(C)), (xmean_new - xmean)) / sigma
            hsig = np.linalg.norm(ps)/np.sqrt(1-(1-self.cs)**(2*evals/self.lambda_))/self.chiN < 1.4 + 2/(self.dim+1)
            pc = (1-self.ccov) * pc + hsig * np.sqrt(self.ccov*(2-self.ccov)*self.mueff) * (xmean_new - xmean) / sigma
            
            C = (1-self.c1-self.cmu) * C + self.c1 * (np.outer(pc, pc) + (1-hsig) * self.ccov*(2-self.ccov) * C)
            
            # More robust covariance update, incorporating rank weights
            for i in range(self.mu):
                y = (x[:, i] - xmean) / sigma
                C += self.cmu * rank_weights[i] * np.outer(y, y)

            # Step-size adaptation based on fitness progress
            if len(performance_history) > self.step_size_adaptation_threshold:
                recent_progress = performance_history[-self.step_size_adaptation_threshold] - performance_history[-1]
                if recent_progress > 0:
                    sigma = sigma * np.exp(0.05 * recent_progress / sigma)  # Increase sigma if progress is good
                else:
                    sigma = sigma * np.exp(0.01 * recent_progress / sigma)  # Decrease sigma if progress is bad

            sigma = sigma * np.exp((self.cs/self.cs_damps) * (np.linalg.norm(ps)/self.chiN - 1))

            xmean = xmean_new

            # Repair covariance matrix (ensure positive definiteness)
            C = np.triu(C) + np.transpose(np.triu(C,1))
            try:
                np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = C + np.eye(self.dim) * 1e-6

            # Orthogonal Subspace Learning
            orthogonal_learning_counter += 1

            # Adaptive orthogonal learning frequency
            adaptive_orthogonal_learning_threshold = int(self.orthogonal_learning_threshold * (1 + 0.2*(f_opt-np.mean([self.lb, self.ub]))))

            if orthogonal_learning_counter > adaptive_orthogonal_learning_threshold:
                orthogonal_learning_counter = 0

                # Calculate the change in xmean
                delta_xmean = xmean_new - xmean

                # Perform SVD on the covariance matrix
                try:
                    U, S, V = np.linalg.svd(C)
                except np.linalg.LinAlgError:
                    U, S, V = np.linalg.svd(C + np.eye(self.dim) * 1e-6)  # Adding small value to diagonal

                # Project delta_xmean onto the principal components
                delta_xmean_projected = np.dot(U.T, delta_xmean)

                # Update xmean along the principal components (only top components)
                num_components_to_use = min(self.dim, 5)  # Limiting to top 5 for stability
                xmean = xmean + np.dot(U[:, :num_components_to_use], delta_xmean_projected[:num_components_to_use])

                xmean = np.clip(xmean, self.lb, self.ub)  # Ensure bounds are respected

            # Archive maintenance (for diversity)
            for i in range(self.lambda_):
                if self.is_diverse(x[:, i]):
                    self.archive_x.append(x[:, i].copy())
                    self.archive_f.append(f[i])
                    if len(self.archive_x) > self.archive_size:
                        # Remove the oldest entry (FIFO)
                        self.archive_x.pop(0)
                        self.archive_f.pop(0)

            # Restart mechanism
            if np.max(np.diag(C)) > (10**7) * sigma or stagnation_counter > self.stagnation_threshold:  # Aggressive restart
                restart_iter += 1
                #Option 1: Restart from a random archive member
                if len(self.archive_x) > 0 and np.random.rand() < 0.5:
                    idx = np.random.randint(0, len(self.archive_x))
                    xmean = self.archive_x[idx].copy()
                else: #Option 2: Restart from a completely random position
                    xmean = np.random.uniform(self.lb, self.ub, size=self.dim)
                
                sigma = self.initial_sigma  # Reset sigma
                C = np.eye(self.dim)
                pc = np.zeros(self.dim)
                ps = np.zeros(self.dim)
                stagnation_counter = 0  # reset stagnation
                orthogonal_learning_counter = 0  # reset orthogonal learning counter
                self.lambda_ = int(4 + np.floor(3 * np.log(self.dim)))  # Reset population size to original
                performance_history = []
                step_size_history = []

                if restart_iter > self.max_restarts:
                    break

            if np.any(np.isnan(C)):
                C = np.eye(self.dim)
                sigma = self.initial_sigma
                pc = np.zeros(self.dim)
                ps = np.zeros(self.dim)
                stagnation_counter = 0
                orthogonal_learning_counter = 0
                self.lambda_ = int(4 + np.floor(3 * np.log(self.dim)))  # Reset population size
                performance_history = []
                step_size_history = []
                    
        return f_opt, x_opt
    
    def is_diverse(self, x):
        if not self.archive_x:
            return True
        for archived_x in self.archive_x:
            if np.linalg.norm(x - archived_x) < self.diversity_threshold:
                return False
        return True