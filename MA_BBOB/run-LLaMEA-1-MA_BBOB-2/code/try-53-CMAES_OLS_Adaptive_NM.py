import numpy as np
from scipy.optimize import minimize

class CMAES_OLS_Adaptive_NM:
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
        self.nelder_mead_iterations = 50

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
        performance_history = [] #Track last 'n' best fitness values

        while evals < self.budget:
            # Generate and evaluate offspring
            z = np.random.normal(0, 1, size=(self.dim, self.lambda_))
            y = np.dot(np.linalg.cholesky(C), z)
            x = xmean.reshape(-1, 1) + sigma * y
            x = np.clip(x, self.lb, self.ub)
            
            f = np.array([func(x[:,i]) if evals + i < self.budget else np.inf for i in range(self.lambda_)])
            evals += self.lambda_
            
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

                # Adaptive Lambda: Increase lambda if consistently improving
                if len(performance_history) > 5 and all(performance_history[i] > performance_history[i+1] for i in range(len(performance_history)-1)):
                     self.lambda_ = int(self.lambda_ * 1.1) #Increase population size by 10%
                     self.mu = int(self.lambda_ * self.mu_ratio)  # Update mu
                     self.weights = np.log(self.mu + 1/2) - np.log(np.arange(1, self.mu + 1))  # Weights for recombination
                     self.weights = self.weights / np.sum(self.weights)
                     self.mueff = np.sum(self.weights)**2 / np.sum(self.weights**2)
                     self.c1 = self.ccov / ((self.dim+1.3)**2 + self.mueff)
                     self.cmu = min(1-self.c1, self.ccov * (self.mueff-2+1/self.mueff) / ((self.dim+2.0)**2 + self.mueff))

            else:
                stagnation_counter += 1
                if len(performance_history) > 0:
                    performance_history.pop(0) #Remove oldest entry


            # Update distribution parameters
            xmean_new = np.sum(x[:, :self.mu] * self.weights, axis=1)
            
            ps = (1-self.cs) * ps + np.sqrt(self.cs*(2-self.cs)*self.mueff) * np.dot(np.linalg.inv(np.linalg.cholesky(C)), (xmean_new - xmean)) / sigma
            hsig = np.linalg.norm(ps)/np.sqrt(1-(1-self.cs)**(2*evals/self.lambda_))/self.chiN < 1.4 + 2/(self.dim+1)
            pc = (1-self.ccov) * pc + hsig * np.sqrt(self.ccov*(2-self.ccov)*self.mueff) * (xmean_new - xmean) / sigma
            
            C = (1-self.c1-self.cmu) * C + self.c1 * (np.outer(pc, pc) + (1-hsig) * self.ccov*(2-self.ccov) * C)
            
            # More robust covariance update
            for i in range(self.mu):
                y = (x[:, i] - xmean) / sigma
                C += self.cmu * self.weights[i] * np.outer(y, y)

            sigma = sigma * np.exp((self.cs/self.cs_damps) * (np.linalg.norm(ps)/self.chiN - 1))
            
            xmean = xmean_new

            # Repair covariance matrix (ensure positive definiteness)
            C = np.triu(C) + np.transpose(np.triu(C,1))
            try:
                np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = C + np.eye(self.dim) * 1e-6

            # Orthogonal Subspace Learning
            orthogonal_learning_counter +=1

            #Adaptive orthogonal learning frequency
            adaptive_orthogonal_learning_threshold = int(self.orthogonal_learning_threshold * (1 + 0.2*(f_opt-np.mean(func.bounds.ub))))

            if orthogonal_learning_counter > adaptive_orthogonal_learning_threshold:
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

            # Restart mechanism
            if np.max(np.diag(C)) > (10**7) * sigma or stagnation_counter > self.stagnation_threshold: # Aggressive restart
                restart_iter += 1

                # Local Search with Nelder-Mead
                if evals + self.nelder_mead_iterations < self.budget:
                    bounds = [(self.lb, self.ub) for _ in range(self.dim)]
                    res = minimize(func, x_opt, method='Nelder-Mead', bounds=bounds, options={'maxiter': self.nelder_mead_iterations, 'maxfev': self.nelder_mead_iterations})
                    if res.fun < f_opt:
                        f_opt = res.fun
                        x_opt = res.x
                    evals += res.nfev

                xmean = np.random.uniform(self.lb, self.ub, size=self.dim)
                sigma = self.initial_sigma # Reset sigma
                C = np.eye(self.dim)
                pc = np.zeros(self.dim)
                ps = np.zeros(self.dim)
                stagnation_counter = 0 #reset stagnation
                orthogonal_learning_counter = 0 #reset orthogonal learning counter
                self.lambda_ = int(4 + np.floor(3 * np.log(self.dim)))  # Reset population size to original
                performance_history = []

                if restart_iter > self.max_restarts:
                    break

            if np.any(np.isnan(C)):
                C = np.eye(self.dim)
                sigma = self.initial_sigma
                pc = np.zeros(self.dim)
                ps = np.zeros(self.dim)
                stagnation_counter = 0
                orthogonal_learning_counter = 0
                self.lambda_ = int(4 + np.floor(3 * np.log(self.dim))) # Reset population size
                performance_history = []
                    
        return f_opt, x_opt