import numpy as np

class CMAES_OLS_Adaptive:
    def __init__(self, budget=10000, dim=10, mu_ratio=0.25, cs=0.3, dsigma=0.2, ccov=0.3, initial_sigma=0.5):
        self.budget = budget
        self.dim = dim
        self.mu_ratio = mu_ratio
        self.cs = cs
        self.dsigma = dsigma
        self.ccov = ccov
        self.lb = -5.0
        self.ub = 5.0
        self.initial_sigma = initial_sigma
        self.population_scaling = 1.0
        self.ols_frequency = 10 # Initial orthogonal learning frequency
        self.decay_factor = 0.99  # Decay factor for orthogonal move size
        self.min_orthogonal_move_size = 0.001
        self.success_history_length = 10
        self.success_history = [] # Track success history of orthogonal moves
        self.learning_rate_scaling = 1.0 #Dynamic scaling of learning rate
        self.step_size_adaptation_rate = 1.0 # Adapt step size based on performance
        self.exploration_exploitation_tradeoff = 0.5 # Dynamic tradeoff
        self.stagnation_threshold = 50
        self.restart_iter = 0
        self.max_restarts = 5
        self.C_decay_rate = 0.999 #Decay rate to prevent C from exploding


    def __call__(self, func):
        # Initialize variables
        xmean = np.random.uniform(self.lb, self.ub, size=self.dim)  # Initial guess of mean
        sigma = self.initial_sigma  # Overall step size

        lambda_ = int(4 + np.floor(3 * np.log(self.dim) * self.population_scaling))  # Population size
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

        stagnation_counter = 0
        orthogonal_learning_counter = 0
        orthogonal_move_size = 0.1 # Initial size for orthogonal moves

        def is_C_valid(C):
            try:
                np.linalg.cholesky(C)
                return True
            except np.linalg.LinAlgError:
                return False

        while evals < self.budget:
            # Generate and evaluate offspring
            z = np.random.normal(0, 1, size=(self.dim, lambda_))
            y = np.dot(np.linalg.cholesky(C), z)
            x = xmean.reshape(-1, 1) + sigma * y
            x = np.clip(x, self.lb, self.ub)

            f = np.array([func(x[:, i]) if evals + i < self.budget else np.inf for i in range(lambda_)])
            evals += lambda_

            # Sort by fitness
            idx = np.argsort(f)
            f = f[idx]
            x = x[:, idx]

            # Update optimal solution
            if f[0] < f_opt:
                f_opt = f[0]
                x_opt = x[:, 0].copy()  # Ensure x_opt is a copy
                stagnation_counter = 0  # Reset stagnation counter
            else:
                stagnation_counter += 1

            # Update distribution parameters
            xmean_new = np.sum(x[:, :mu] * weights, axis=1)

            ps = (1-cs) * ps + np.sqrt(cs*(2-cs)*mueff) * np.dot(np.linalg.inv(np.linalg.cholesky(C)), (xmean_new - xmean)) / sigma
            hsig = np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*evals/lambda_))/chiN < 1.4 + 2/(self.dim+1)
            pc = (1-ccov) * pc + hsig * np.sqrt(ccov*(2-ccov)*mueff) * (xmean_new - xmean) / sigma

            # Simplified rank-one update to avoid instabilities
            y = (xmean_new - xmean) / sigma
            C = (1-c1) * C + c1 * np.outer(pc, pc)  # Use only pc for covariance update

            sigma = sigma * np.exp((cs/damps) * (np.linalg.norm(ps)/chiN - 1)) * self.step_size_adaptation_rate

            xmean = xmean_new

            # Repair covariance matrix (ensure positive definiteness)
            C = np.triu(C) + np.transpose(np.triu(C,1))
            C = C * self.C_decay_rate

            # Spectral correction
            try:
                S, U = np.linalg.eig(C)
                S = np.real(S)
                U = np.real(U)
                S[S < 0] = 1e-6  # Ensure eigenvalues are positive
                C = U @ np.diag(S) @ U.T
            except np.linalg.LinAlgError:
                C = C + np.eye(self.dim) * 1e-6

            if not is_C_valid(C):
                C = C + np.eye(self.dim) * 1e-6

            # Orthogonal Subspace Learning
            orthogonal_learning_counter += 1
            if orthogonal_learning_counter > self.ols_frequency:
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
                num_components_to_use = min(self.dim, int(self.dim * self.exploration_exploitation_tradeoff))  # Adaptive number of components
                xmean = xmean + self.learning_rate_scaling * np.dot(U[:, :num_components_to_use], delta_xmean_projected[:num_components_to_use])

                xmean = np.clip(xmean, self.lb, self.ub)  # Ensure bounds are respected

                #Orthogonal move
                orthogonal_direction = np.random.normal(0, 1, size=self.dim)
                orthogonal_direction -= np.dot(orthogonal_direction, delta_xmean) * delta_xmean / np.linalg.norm(delta_xmean)**2
                orthogonal_direction /= np.linalg.norm(orthogonal_direction)

                xmean_orthogonal = xmean + orthogonal_move_size * orthogonal_direction
                xmean_orthogonal = np.clip(xmean_orthogonal, self.lb, self.ub)

                f_orthogonal = func(xmean_orthogonal) if evals + 1 < self.budget else np.inf
                evals += 1
                
                if f_orthogonal < f_opt:
                    f_opt = f_orthogonal
                    x_opt = xmean_orthogonal.copy()
                    xmean = xmean_orthogonal.copy()
                    stagnation_counter = 0
                    orthogonal_move_size *= 1.1  # Increase orthogonal move size if successful
                    self.learning_rate_scaling *= 1.05 # Increase learning rate if successful
                    self.success_history.append(1)
                else:
                    orthogonal_move_size *= self.decay_factor  # Decrease orthogonal move size if unsuccessful
                    self.learning_rate_scaling *= 0.95 # Decrease learning rate if unsuccessful
                    self.success_history.append(0)

                orthogonal_move_size = max(orthogonal_move_size, self.min_orthogonal_move_size)  # Ensure orthogonal move size doesn't vanish
                self.learning_rate_scaling = np.clip(self.learning_rate_scaling, 0.5, 2.0) # Clip scaling factor
                
                # Adapt orthogonal learning frequency
                if f_orthogonal < f_opt:
                   self.ols_frequency = max(1, int(self.ols_frequency * 0.9))
                else:
                   self.ols_frequency = min(100, int(self.ols_frequency * 1.1))

                # Maintain success history
                if len(self.success_history) > self.success_history_length:
                    self.success_history.pop(0)

                # Adapt step size based on success history
                success_rate = np.mean(self.success_history) if self.success_history else 0.5
                self.step_size_adaptation_rate = 1 + 0.2 * (success_rate - 0.5)
                self.step_size_adaptation_rate = np.clip(self.step_size_adaptation_rate, 0.5, 1.5)

                # Adapt exploration-exploitation tradeoff based on stagnation and success
                if stagnation_counter > self.stagnation_threshold / 2:
                    self.exploration_exploitation_tradeoff = min(1.0, self.exploration_exploitation_tradeoff + 0.05) # Increase exploration
                else:
                    self.exploration_exploitation_tradeoff = max(0.1, self.exploration_exploitation_tradeoff - 0.02) # Increase exploitation


            # Restart mechanism
            if np.max(np.diag(C)) > (10**7) * sigma or stagnation_counter > self.stagnation_threshold:  # Aggressive restart
                self.restart_iter += 1
                xmean = np.random.uniform(self.lb, self.ub, size=self.dim)
                sigma = self.initial_sigma  # Reset sigma
                C = np.eye(self.dim)
                pc = np.zeros(self.dim)
                ps = np.zeros(self.dim)
                stagnation_counter = 0  # reset stagnation
                orthogonal_learning_counter = 0  # reset orthogonal learning counter
                orthogonal_move_size = 0.1
                self.learning_rate_scaling = 1.0
                self.ols_frequency = 10
                self.success_history = []
                self.step_size_adaptation_rate = 1.0
                self.exploration_exploitation_tradeoff = 0.5
                self.population_scaling = np.clip(self.population_scaling * 0.9, 0.5, 1.5) #reduce population slightly
                self.C_decay_rate = 0.999


                lambda_ = int(4 + np.floor(3 * np.log(self.dim) * self.population_scaling))  # Population size
                mu = int(lambda_ * self.mu_ratio)  # Number of parents

                if self.restart_iter > self.max_restarts:
                    break

            if np.any(np.isnan(C)):
                C = np.eye(self.dim)
                sigma = self.initial_sigma
                pc = np.zeros(self.dim)
                ps = np.zeros(self.dim)
                stagnation_counter = 0
                orthogonal_learning_counter = 0
                orthogonal_move_size = 0.1
                self.learning_rate_scaling = 1.0
                self.ols_frequency = 10
                self.success_history = []
                self.step_size_adaptation_rate = 1.0
                self.exploration_exploitation_tradeoff = 0.5
                self.population_scaling = 1.0
                self.C_decay_rate = 0.999


        return f_opt, x_opt