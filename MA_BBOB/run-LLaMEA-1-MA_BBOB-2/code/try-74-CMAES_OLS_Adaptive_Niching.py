import numpy as np

class CMAES_OLS_Adaptive_Niching:
    def __init__(self, budget=10000, dim=10, mu_ratio=0.25, cs=0.3, dsigma=0.2, ccov=0.3, initial_sigma=0.5, num_niches=3):
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
        self.ols_frequency = 10  # Initial orthogonal learning frequency
        self.num_niches = num_niches
        self.niches = []

    def initialize_niche(self):
        xmean = np.random.uniform(self.lb, self.ub, size=self.dim)
        sigma = self.initial_sigma
        lambda_ = int(4 + np.floor(3 * np.log(self.dim) * self.population_scaling))
        mu = int(lambda_ * self.mu_ratio)

        weights = np.log(mu + 1/2) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = np.sum(weights)**2 / np.sum(weights**2)

        C = np.eye(self.dim)
        pc = np.zeros(self.dim)
        ps = np.zeros(self.dim)
        chiN = np.sqrt(self.dim) * (1 - (1/(4*self.dim)) + 1/(12*self.dim**2))

        cs = self.cs
        damps = 1 + self.dsigma * max(0, np.sqrt((mueff-1)/(self.dim+1))-1) + cs
        ccov = self.ccov
        c1 = ccov / ((self.dim+1.3)**2 + mueff)
        cmu = min(1-c1, ccov * (mueff-2+1/mueff) / ((self.dim+2.0)**2 + mueff))

        return {
            'xmean': xmean,
            'sigma': sigma,
            'lambda_': lambda_,
            'mu': mu,
            'weights': weights,
            'mueff': mueff,
            'C': C,
            'pc': pc,
            'ps': ps,
            'chiN': chiN,
            'cs': cs,
            'damps': damps,
            'ccov': ccov,
            'c1': c1,
            'cmu': cmu,
            'stagnation_counter': 0,
            'orthogonal_learning_counter': 0,
            'orthogonal_move_size': 0.1,
            'learning_rate_scaling': 1.0,
            'ols_frequency': 10,
            'evals': 0
        }

    def __call__(self, func):
        # Initialize niches
        self.niches = [self.initialize_niche() for _ in range(self.num_niches)]

        f_opt = np.Inf
        x_opt = None
        evals = 0

        stagnation_threshold = 50
        max_restarts = 3 # Reduced restarts per niche
        
        def is_C_valid(C):
            try:
                np.linalg.cholesky(C)
                return True
            except np.linalg.LinAlgError:
                return False

        while evals < self.budget:
            for i, niche in enumerate(self.niches):
                if evals >= self.budget:
                    break  # Ensure budget is not exceeded

                # Generate and evaluate offspring
                z = np.random.normal(0, 1, size=(self.dim, niche['lambda_']))
                y = np.dot(np.linalg.cholesky(niche['C']), z)
                x = niche['xmean'].reshape(-1, 1) + niche['sigma'] * y
                x = np.clip(x, self.lb, self.ub)

                f = np.array([func(x[:, j]) if evals + j < self.budget else np.inf for j in range(niche['lambda_'])])
                evals += niche['lambda_']
                niche['evals'] += niche['lambda_']

                # Sort by fitness
                idx = np.argsort(f)
                f = f[idx]
                x = x[:, idx]

                # Update optimal solution
                if f[0] < f_opt:
                    f_opt = f[0]
                    x_opt = x[:, 0].copy()

                # Update distribution parameters
                xmean_new = np.sum(x[:, :niche['mu']] * niche['weights'], axis=1)

                niche['ps'] = (1-niche['cs']) * niche['ps'] + np.sqrt(niche['cs']*(2-niche['cs'])*niche['mueff']) * np.dot(np.linalg.inv(np.linalg.cholesky(niche['C'])), (xmean_new - niche['xmean'])) / niche['sigma']
                hsig = np.linalg.norm(niche['ps'])/np.sqrt(1-(1-niche['cs'])**(2*niche['evals']/niche['lambda_']))/niche['chiN'] < 1.4 + 2/(self.dim+1)
                niche['pc'] = (1-niche['ccov']) * niche['pc'] + hsig * np.sqrt(niche['ccov']*(2-niche['ccov'])*niche['mueff']) * (xmean_new - niche['xmean']) / niche['sigma']

                # Simplified rank-one update
                y = (xmean_new - niche['xmean']) / niche['sigma']
                niche['C'] = (1-niche['c1']) * niche['C'] + niche['c1'] * np.outer(niche['pc'], niche['pc'])

                niche['sigma'] = niche['sigma'] * np.exp((niche['cs']/niche['damps']) * (np.linalg.norm(niche['ps'])/niche['chiN'] - 1))

                # Repair covariance matrix
                niche['C'] = np.triu(niche['C']) + np.transpose(np.triu(niche['C'],1))
                if not is_C_valid(niche['C']):
                    niche['C'] = niche['C'] + np.eye(self.dim) * 1e-6

                niche['xmean'] = xmean_new

                 # Stagnation check and orthogonal learning inside the loop
                if f[0] < f_opt:
                    niche['stagnation_counter'] = 0
                else:
                    niche['stagnation_counter'] += 1
                    
                # Orthogonal Subspace Learning
                niche['orthogonal_learning_counter'] += 1
                if niche['orthogonal_learning_counter'] > niche['ols_frequency']:
                    niche['orthogonal_learning_counter'] = 0

                    # Calculate the change in xmean
                    delta_xmean = xmean_new - niche['xmean']

                    # Perform SVD on the covariance matrix
                    try:
                        U, S, V = np.linalg.svd(niche['C'])
                    except np.linalg.LinAlgError:
                        U, S, V = np.linalg.svd(niche['C'] + np.eye(self.dim) * 1e-6)  # Adding small value to diagonal

                    # Project delta_xmean onto the principal components
                    delta_xmean_projected = np.dot(U.T, delta_xmean)

                    # Update xmean along the principal components (only top components)
                    num_components_to_use = min(self.dim, 5)  # Limiting to top 5 for stability
                    niche['xmean'] = niche['xmean'] + niche['learning_rate_scaling'] * np.dot(U[:, :num_components_to_use], delta_xmean_projected[:num_components_to_use])

                    niche['xmean'] = np.clip(niche['xmean'], self.lb, self.ub)  # Ensure bounds are respected

                    # Orthogonal move
                    orthogonal_direction = np.random.normal(0, 1, size=self.dim)
                    orthogonal_direction -= np.dot(orthogonal_direction, delta_xmean) * delta_xmean / np.linalg.norm(delta_xmean)**2
                    orthogonal_direction /= np.linalg.norm(orthogonal_direction)

                    xmean_orthogonal = niche['xmean'] + niche['orthogonal_move_size'] * orthogonal_direction
                    xmean_orthogonal = np.clip(xmean_orthogonal, self.lb, self.ub)

                    f_orthogonal = func(xmean_orthogonal) if evals + 1 < self.budget else np.inf
                    evals += 1
                    niche['evals'] += 1

                    if f_orthogonal < f_opt:
                        f_opt = f_orthogonal
                        x_opt = xmean_orthogonal.copy()
                        niche['xmean'] = xmean_orthogonal.copy()
                        niche['stagnation_counter'] = 0
                        niche['orthogonal_move_size'] *= 1.1  # Increase orthogonal move size if successful
                        niche['learning_rate_scaling'] *= 1.05  # Increase learning rate if successful
                    else:
                        niche['orthogonal_move_size'] *= 0.9  # Decrease orthogonal move size if unsuccessful
                        niche['learning_rate_scaling'] *= 0.95  # Decrease learning rate if unsuccessful

                    niche['orthogonal_move_size'] = np.clip(niche['orthogonal_move_size'], 0.01, 1.0)  # Clip orthogonal move size
                    niche['learning_rate_scaling'] = np.clip(niche['learning_rate_scaling'], 0.5, 2.0)  # Clip scaling factor

                    # Adapt orthogonal learning frequency
                    if f_orthogonal < f_opt:
                        niche['ols_frequency'] = max(1, int(niche['ols_frequency'] * 0.9))
                    else:
                        niche['ols_frequency'] = min(100, int(niche['ols_frequency'] * 1.1))

                # Restart mechanism for each niche
                if np.max(np.diag(niche['C'])) > (10**7) * niche['sigma'] or niche['stagnation_counter'] > stagnation_threshold:
                    # Reduced restarts for individual niches
                    if niche['stagnation_counter'] > stagnation_threshold:
                       restart_iter = 0 # Reset for each individual stagnation
                    else:
                       restart_iter += 1

                    niche['xmean'] = np.random.uniform(self.lb, self.ub, size=self.dim)
                    niche['sigma'] = self.initial_sigma
                    niche['C'] = np.eye(self.dim)
                    niche['pc'] = np.zeros(self.dim)
                    niche['ps'] = np.zeros(self.dim)
                    niche['stagnation_counter'] = 0
                    niche['orthogonal_learning_counter'] = 0
                    niche['orthogonal_move_size'] = 0.1
                    niche['learning_rate_scaling'] = 1.0
                    niche['ols_frequency'] = 10

                    if restart_iter > max_restarts:
                        self.niches[i] = self.initialize_niche() # Reinitialize if exceeding max restarts

                if np.any(np.isnan(niche['C'])):
                    niche['C'] = np.eye(self.dim)
                    niche['sigma'] = self.initial_sigma
                    niche['pc'] = np.zeros(self.dim)
                    niche['ps'] = np.zeros(self.dim)
                    niche['stagnation_counter'] = 0
                    niche['orthogonal_learning_counter'] = 0
                    niche['orthogonal_move_size'] = 0.1
                    niche['learning_rate_scaling'] = 1.0
                    niche['ols_frequency'] = 10

        return f_opt, x_opt