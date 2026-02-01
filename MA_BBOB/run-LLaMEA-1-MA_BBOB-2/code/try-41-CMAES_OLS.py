import numpy as np

class CMAES_OLS:
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

    def __call__(self, func):
        # Initialize variables
        xmean = np.random.uniform(self.lb, self.ub, size=self.dim)  # Initial guess of mean
        sigma = self.initial_sigma  # Overall step size

        lambda_ = int(4 + np.floor(3 * np.log(self.dim)))  # Population size
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
        max_restarts = 5 # Increased restarts

        stagnation_counter = 0
        stagnation_threshold = 50 # Check for stagnation every 50 iterations
        
        orthogonal_learning_threshold = 100 # Apply orthogonal learning after this many iterations
        orthogonal_learning_counter = 0
        
        # Adaptive cmu and c1
        cmu_learning_rate = 0.1
        c1_learning_rate = 0.1

        while evals < self.budget:
            # Generate and evaluate offspring
            z = np.random.normal(0, 1, size=(self.dim, lambda_))
            y = np.dot(np.linalg.cholesky(C), z)
            x = xmean.reshape(-1, 1) + sigma * y
            x = np.clip(x, self.lb, self.ub)
            
            f = np.array([func(x[:,i]) if evals + i < self.budget else np.inf for i in range(lambda_)])
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
            try:
                np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = C + np.eye(self.dim) * 1e-6

            # Orthogonal Subspace Learning
            orthogonal_learning_counter +=1
            if orthogonal_learning_counter > orthogonal_learning_threshold:
                orthogonal_learning_counter = 0

                # Calculate the change in xmean
                delta_xmean = xmean_new - xmean

                # Perform SVD on the covariance matrix
                try:
                    U, S, V = np.linalg.svd(C)
                except np.linalg.LinAlgError:
                    U, S, V = np.linalg.svd(C + np.eye(self.dim) * 1e-6) #Adding small value to diagonal

                # Project delta_xmean onto the principal components
                delta_xmean_projected = np.dot(U.T, delta_xmean)

                # Update xmean along the principal components (only top components)
                num_components_to_use = min(self.dim, 5)  # Limiting to top 5 for stability
                xmean = xmean + np.dot(U[:, :num_components_to_use], delta_xmean_projected[:num_components_to_use])

                xmean = np.clip(xmean, self.lb, self.ub)  # Ensure bounds are respected
                
                # Adapt cmu and c1 based on the success of OSL
                if f[0] < f_opt:
                    cmu *= (1 + cmu_learning_rate)
                    c1 *= (1 - c1_learning_rate)
                else:
                    cmu *= (1 - cmu_learning_rate)
                    c1 *= (1 + c1_learning_rate)
                    
                cmu = max(0, min(cmu, 1))
                c1 = max(0, min(c1, 1))


            # Restart mechanism
            condition_number = np.max(S) / np.min(S) if np.min(S) > 0 else np.inf # Condition number
            flat_fitness = np.std(f) < 1e-12 #Stagnation by fitness
            
            if np.max(np.diag(C)) > (10**7) * sigma or stagnation_counter > stagnation_threshold or condition_number > 1e14 or flat_fitness: # Aggressive restart
                restart_iter += 1
                xmean = np.random.uniform(self.lb, self.ub, size=self.dim)
                sigma = self.initial_sigma # Reset sigma
                C = np.eye(self.dim)
                pc = np.zeros(self.dim)
                ps = np.zeros(self.dim)
                stagnation_counter = 0 #reset stagnation
                orthogonal_learning_counter = 0 #reset orthogonal learning counter

                if restart_iter > max_restarts:
                    break

            if np.any(np.isnan(C)):
                C = np.eye(self.dim)
                sigma = self.initial_sigma
                pc = np.zeros(self.dim)
                ps = np.zeros(self.dim)
                stagnation_counter = 0
                orthogonal_learning_counter = 0
                    
        return f_opt, x_opt