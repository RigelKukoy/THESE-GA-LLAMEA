import numpy as np

class CMAES_OLS_Adaptive_Gradient_Momentum:
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
        self.gradient_estimation_samples = 5
        self.momentum_factor = 0.1 # Momentum for orthogonal direction

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

        restart_iter = 0
        max_restarts = 5  # Increased restarts

        stagnation_counter = 0
        stagnation_threshold = 50  # Check for stagnation every 50 iterations

        orthogonal_learning_threshold = 100  # Apply orthogonal learning after this many iterations
        orthogonal_learning_counter = 0
        orthogonal_move_size = 0.1 # Initial size for orthogonal moves
        
        learning_rate_scaling = 1.0 #Dynamic scaling of learning rate

        gradient_estimation_size = 0.001 # Step size for gradient estimation
        
        # Momentum for orthogonal direction
        self.orthogonal_direction_momentum = np.zeros(self.dim) 

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

            C = (1-c1-cmu) * C + c1 * (np.outer(pc, pc) + (1-hsig) * ccov*(2-ccov) * C)

            # More robust covariance update
            for i in range(mu):
                y = (x[:, i] - xmean) / sigma
                C += cmu * weights[i] * np.outer(y, y)

            sigma = sigma * np.exp((cs/damps) * (np.linalg.norm(ps)/chiN - 1))

            xmean = xmean_new

            # Repair covariance matrix (ensure positive definiteness)
            C = np.triu(C) + np.transpose(np.triu(C,1))
            if not is_C_valid(C):
                C = C + np.eye(self.dim) * 1e-6

            # Orthogonal Subspace Learning
            orthogonal_learning_counter += 1
            if orthogonal_learning_counter > orthogonal_learning_threshold:
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
                xmean = xmean + learning_rate_scaling * np.dot(U[:, :num_components_to_use], delta_xmean_projected[:num_components_to_use])

                xmean = np.clip(xmean, self.lb, self.ub)  # Ensure bounds are respected

                #Orthogonal move based on gradient estimation
                gradient = np.zeros(self.dim)
                f_xmean = func(xmean) if evals + 1 <= self.budget else np.inf
                evals += 1

                for _ in range(self.gradient_estimation_samples):
                    direction = np.random.normal(0, 1, size=self.dim)
                    direction /= np.linalg.norm(direction)

                    x_plus = xmean + gradient_estimation_size * direction
                    x_minus = xmean - gradient_estimation_size * direction

                    x_plus = np.clip(x_plus, self.lb, self.ub)
                    x_minus = np.clip(x_minus, self.lb, self.ub)

                    f_plus = func(x_plus) if evals + 1 <= self.budget else np.inf
                    evals += 1

                    f_minus = func(x_minus) if evals + 1 <= self.budget else np.inf
                    evals += 1

                    gradient += (f_plus - f_minus) / (2 * gradient_estimation_size) * direction
                
                gradient /= self.gradient_estimation_samples

                orthogonal_direction = np.random.normal(0, 1, size=self.dim)
                orthogonal_direction -= np.dot(orthogonal_direction, gradient) * gradient / np.linalg.norm(gradient)**2
                orthogonal_direction /= np.linalg.norm(orthogonal_direction)
                
                # Apply momentum to the orthogonal direction
                self.orthogonal_direction_momentum = (self.momentum_factor * self.orthogonal_direction_momentum +
                                                       (1 - self.momentum_factor) * orthogonal_direction)
                orthogonal_direction = self.orthogonal_direction_momentum / np.linalg.norm(self.orthogonal_direction_momentum)

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
                    learning_rate_scaling *= 1.05 # Increase learning rate if successful
                else:
                    orthogonal_move_size *= 0.9  # Decrease orthogonal move size if unsuccessful
                    learning_rate_scaling *= 0.95 # Decrease learning rate if unsuccessful

                orthogonal_move_size = np.clip(orthogonal_move_size, 0.01, 1.0) # Clip orthogonal move size
                learning_rate_scaling = np.clip(learning_rate_scaling, 0.5, 2.0) # Clip scaling factor


            # Restart mechanism
            if np.max(np.diag(C)) > (10**7) * sigma or stagnation_counter > stagnation_threshold:  # Aggressive restart
                restart_iter += 1
                xmean = np.random.uniform(self.lb, self.ub, size=self.dim)
                sigma = self.initial_sigma  # Reset sigma
                C = np.eye(self.dim)

                # Adapt covariance matrix using the history of successful steps
                if restart_iter > 1 and f_opt < np.inf:  # only apply after the first restart and if a valid solution was found
                    C = np.diag(np.var(x[:, :mu], axis=1)) #Use variance of the best solutions of the last iteration

                pc = np.zeros(self.dim)
                ps = np.zeros(self.dim)
                stagnation_counter = 0  # reset stagnation
                orthogonal_learning_counter = 0  # reset orthogonal learning counter
                orthogonal_move_size = 0.1
                learning_rate_scaling = 1.0
                self.orthogonal_direction_momentum = np.zeros(self.dim)
                
                self.population_scaling = np.clip(self.population_scaling * 0.9, 0.5, 1.5) #reduce population slightly

                lambda_ = int(4 + np.floor(3 * np.log(self.dim) * self.population_scaling))  # Population size
                mu = int(lambda_ * self.mu_ratio)  # Number of parents

                if restart_iter > max_restarts:
                    break

            if np.any(np.isnan(C)):
                C = np.eye(self.dim)
                sigma = self.initial_sigma
                pc = np.zeros(self.dim)
                ps = np.zeros(self.dim)
                stagnation_counter = 0
                orthogonal_learning_counter = 0
                orthogonal_move_size = 0.1
                learning_rate_scaling = 1.0
                self.orthogonal_direction_momentum = np.zeros(self.dim)
                self.population_scaling = 1.0

        return f_opt, x_opt