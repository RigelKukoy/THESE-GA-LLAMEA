import numpy as np

class RingTopologyCMAES:
    def __init__(self, budget=10000, dim=10, num_populations=5, pop_size=None, sigma0=0.5, cs=0.2, cmu=0.3, c_cov=0.1, adaptation_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.num_populations = num_populations
        self.sigma0 = sigma0
        self.cs = cs
        self.cmu = cmu
        self.c_cov = c_cov
        self.adaptation_rate = adaptation_rate
        if pop_size is None:
            self.pop_size = int(budget / num_populations / 10)  # Automatic population size determination
        else:
            self.pop_size = pop_size
        self.mu = self.pop_size // 2
        self.populations = []
        self.best_fitnesses = [np.Inf] * num_populations
        self.best_solutions = [None] * num_populations
        self.evals_per_population = budget // num_populations

        for i in range(num_populations):
            self.populations.append(self._create_population())

    def _create_population(self):
        return {
            'mean': None,
            'sigma': self.sigma0,
            'C': None,
            'ps': None,
            'pc': None,
            'fitness_history': [],
            'restart_criterion': 1e-12,
            'mutation_strength': 1.0 # Initialize mutation strength
        }

    def __call__(self, func):
        total_evals = 0
        global_f_opt = np.Inf
        global_x_opt = None

        for i in range(self.num_populations):
            population = self.populations[i]
            population['mean'] = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
            population['C'] = np.eye(self.dim)
            population['ps'] = np.zeros(self.dim)
            population['pc'] = np.zeros(self.dim)
            evals = 0

            while evals < self.evals_per_population:
                # 1. Sample offspring
                z = np.random.multivariate_normal(np.zeros(self.dim), population['C'], size=self.pop_size)
                x = population['mean'] + population['sigma'] * population['mutation_strength'] * z

                # Clip to bounds
                x = np.clip(x, func.bounds.lb, func.bounds.ub)

                f = np.array([func(xi) for xi in x])
                evals += self.pop_size
                total_evals += self.pop_size

                # 2. Selection and Recombination
                idx = np.argsort(f)
                x_best = x[idx[:self.mu]]
                z_best = z[idx[:self.mu]]

                if np.min(f) < self.best_fitnesses[i]:
                    self.best_fitnesses[i] = np.min(f)
                    self.best_solutions[i] = x[idx[0]]

                # 3. Update mean
                mean_diff = np.mean(z_best, axis=0)
                population['pc'] = (1 - self.cs) * population['pc'] + np.sqrt(self.cs * (2 - self.cs)) * mean_diff
                population['mean'] = population['mean'] + self.cmu * population['sigma'] * population['mutation_strength'] * population['pc']

                # 4. Update covariance matrix
                population['ps'] = (1 - self.c_cov) * population['ps'] + np.sqrt(self.c_cov * (2 - self.c_cov)) * mean_diff
                population['C'] = (1 - self.c_cov) * population['C'] + self.c_cov * (np.outer(population['ps'], population['ps']) - population['C'])

                # 5. Adaptive Variance Scaling (sigma)
                population['sigma'] *= np.exp((self.cs / 2) * (np.linalg.norm(population['ps'])**2 / self.dim - 1))

                # 6. Self-adaptive mutation strength based on population diversity
                distances = np.linalg.norm(x - population['mean'], axis=1)
                diversity = np.std(distances)
                if diversity > 0.1 * population['sigma']: # If population is diverse
                    population['mutation_strength'] *= (1 + self.adaptation_rate) # Increase exploration
                else:
                    population['mutation_strength'] *= (1 - self.adaptation_rate) # Decrease exploration

                population['mutation_strength'] = np.clip(population['mutation_strength'], 0.1, 10) # Keep mutation strength within reasonable bounds
                population['fitness_history'].append(np.min(f))

                # Check for covariance matrix deterioration
                if np.linalg.det(population['C']) < population['restart_criterion'] or not np.all(np.linalg.eigvals(population['C']) > 0):
                    population['C'] = np.eye(self.dim)  # restart C
                    population['sigma'] = self.sigma0  # Reset sigma
                    population['ps'] = np.zeros(self.dim)
                    population['pc'] = np.zeros(self.dim)
                    population['fitness_history'] = []

                # Cooperation: Ring Topology Information Sharing
                if evals % (self.evals_per_population // 2) == 0 and self.num_populations > 1:
                    # Send to the next population in the ring
                    next_index = (i + 1) % self.num_populations
                    self.populations[next_index]['mean'] = 0.7 * self.populations[next_index]['mean'] + 0.3 * population['mean']  # Move towards neighbor's mean
                    # Receive from the previous population in the ring
                    prev_index = (i - 1) % self.num_populations
                    population['mean'] = 0.7 * population['mean'] + 0.3 * self.populations[prev_index]['mean']
            
            # Update global best
            if self.best_fitnesses[i] < global_f_opt:
                global_f_opt = self.best_fitnesses[i]
                global_x_opt = self.best_solutions[i]

        return global_f_opt, global_x_opt