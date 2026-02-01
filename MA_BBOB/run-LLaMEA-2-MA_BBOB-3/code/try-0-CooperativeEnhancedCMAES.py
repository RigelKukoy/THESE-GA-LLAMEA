import numpy as np

class CooperativeEnhancedCMAES:
    def __init__(self, budget=10000, dim=10, num_optimizers=5, mu_percentage=0.25, sigma0=0.5, cs=0.2, cmu=0.3, c_cov=0.1, adaptation_rate=0.1, sharing_interval=50):
        self.budget = budget
        self.dim = dim
        self.num_optimizers = num_optimizers
        self.mu = int(mu_percentage * budget / num_optimizers)
        self.sigma0 = sigma0
        self.cs = cs
        self.cmu = cmu
        self.c_cov = c_cov
        self.adaptation_rate = adaptation_rate
        self.sharing_interval = sharing_interval
        self.optimizers = []
        self.best_fitnesses = [np.Inf] * num_optimizers
        self.best_solutions = [None] * num_optimizers
        self.evals_per_optimizer = budget // num_optimizers  # Ensure budget is divided evenly

        for i in range(num_optimizers):
            self.optimizers.append(self._create_optimizer())

    def _create_optimizer(self):
        return {
            'mean': None,
            'sigma': self.sigma0,
            'C': None,
            'ps': None,
            'pc': None,
            'fitness_history': [],
            'restart_criterion': 1e-12
        }

    def __call__(self, func):
        total_evals = 0
        global_f_opt = np.Inf
        global_x_opt = None

        for i in range(self.num_optimizers):
            optimizer = self.optimizers[i]
            optimizer['mean'] = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
            optimizer['C'] = np.eye(self.dim)
            optimizer['ps'] = np.zeros(self.dim)
            optimizer['pc'] = np.zeros(self.dim)
            evals = 0

            while evals < self.evals_per_optimizer:

                # 1. Sample offspring
                z = np.random.multivariate_normal(np.zeros(self.dim), optimizer['C'], size=self.mu)
                x = optimizer['mean'] + optimizer['sigma'] * z

                # Clip to bounds
                x = np.clip(x, func.bounds.lb, func.bounds.ub)

                f = np.array([func(xi) for xi in x])
                evals += self.mu
                total_evals += self.mu

                # 2. Selection and Recombination
                idx = np.argsort(f)
                x_best = x[idx[:self.mu]]
                z_best = z[idx[:self.mu]]

                if np.min(f) < self.best_fitnesses[i]:
                    self.best_fitnesses[i] = np.min(f)
                    self.best_solutions[i] = x[idx[0]]

                # 3. Update mean
                mean_diff = np.mean(z_best, axis=0)
                optimizer['pc'] = (1 - self.cs) * optimizer['pc'] + np.sqrt(self.cs * (2 - self.cs)) * mean_diff
                optimizer['mean'] = optimizer['mean'] + self.cmu * optimizer['sigma'] * optimizer['pc']

                # 4. Update covariance matrix
                optimizer['ps'] = (1 - self.c_cov) * optimizer['ps'] + np.sqrt(self.c_cov * (2 - self.c_cov)) * mean_diff
                optimizer['C'] = (1 - self.c_cov) * optimizer['C'] + self.c_cov * (np.outer(optimizer['ps'], optimizer['ps']) - optimizer['C'])

                # 5. Adaptive Variance Scaling
                if len(optimizer['fitness_history']) > 1:
                    fitness_improvement = optimizer['fitness_history'][-2] - optimizer['fitness_history'][-1]
                    if fitness_improvement > 0:  # If there is improvement, increase the variance
                        optimizer['sigma'] *= (1 + self.adaptation_rate * fitness_improvement)
                    else:  # Decrease the variance
                        optimizer['sigma'] *= (1 - self.adaptation_rate * abs(fitness_improvement))

                # 6. Update step size (original)
                optimizer['sigma'] *= np.exp((self.cs / 2) * (np.linalg.norm(optimizer['ps'])**2 / self.dim - 1))
                optimizer['fitness_history'].append(np.min(f))

                # Check for covariance matrix deterioration
                if np.linalg.det(optimizer['C']) < optimizer['restart_criterion'] or not np.all(np.linalg.eigvals(optimizer['C']) > 0):
                    optimizer['C'] = np.eye(self.dim)  # restart C
                    optimizer['sigma'] = self.sigma0  # Reset sigma
                    optimizer['ps'] = np.zeros(self.dim)
                    optimizer['pc'] = np.zeros(self.dim)
                    optimizer['fitness_history'] = []

                # Cooperation: intermittent sharing of information
                if evals % self.sharing_interval == 0:
                    best_idx = np.argmin(self.best_fitnesses)
                    best_mean = self.optimizers[best_idx]['mean']
                    optimizer['mean'] = 0.5 * optimizer['mean'] + 0.5 * best_mean  # Move towards best mean

            # Update global best
            if self.best_fitnesses[i] < global_f_opt:
                global_f_opt = self.best_fitnesses[i]
                global_x_opt = self.best_solutions[i]
        
        return global_f_opt, global_x_opt