import numpy as np

class GradientAdaptiveCMAES:
    def __init__(self, budget=10000, dim=10, popsize=None, sigma0=0.5, gradient_samples=5, stepsize_adapt_factor=0.1):
        self.budget = budget
        self.dim = dim
        self.popsize = popsize if popsize is not None else 4 + int(3 * np.log(self.dim))
        self.sigma = sigma0
        self.mean = None
        self.C = None
        self.pc = None
        self.ps = None
        self.chiN = None
        self.eval_count = 0
        self.f_opt = np.inf
        self.x_opt = None
        self.gradient_samples = gradient_samples
        self.stepsize_adapt_factor = stepsize_adapt_factor  # Adjust step size adaptation rate

    def initialize(self, func):
        self.mean = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
        self.C = np.eye(self.dim)
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.chiN = self.dim**0.5 * (1 - 1/(4*self.dim) + 1/(21*self.dim**2))
        self.f_opt = np.inf
        self.x_opt = None

    def sample_population(self):
        z = np.random.normal(0, 1, size=(self.popsize, self.dim))
        A = np.linalg.cholesky(self.C)
        x = self.mean + self.sigma * z @ A.T
        return x

    def estimate_gradient_norm(self, func, x):
        """Estimates the norm of the gradient around a point."""
        gradient_estimates = []
        for _ in range(self.gradient_samples):
            delta = np.random.normal(0, self.sigma, size=self.dim)
            x_plus = np.clip(x + delta, func.bounds.lb, func.bounds.ub)
            x_minus = np.clip(x - delta, func.bounds.lb, func.bounds.ub)

            f_plus = func(x_plus)
            f_minus = func(x_minus)
            
            if not isinstance(f_plus, float) or not isinstance(f_minus, float):
                continue

            gradient_estimates.append((f_plus - f_minus) / (2 * self.sigma))

        if not gradient_estimates:
          return 0.0

        return np.mean(np.abs(np.array(gradient_estimates)))  # Average absolute gradient estimate

    def adapt_step_size(self, gradient_norm):
        """Adapts the step size based on the estimated gradient norm."""
        if gradient_norm > 1:  # Steep slope
            self.sigma *= (1 - self.stepsize_adapt_factor) # Reduce step size
        elif gradient_norm < 0.1:  # Flat region
            self.sigma *= (1 + self.stepsize_adapt_factor)  # Increase step size
        # else: keep step size the same

    def __call__(self, func):
        self.initialize(func)
        mu = self.popsize // 2
        c_sigma = (mu / self.dim) / ((self.dim + 4) + (mu / self.dim))
        c_c = (4 + mu / self.dim) / (self.dim + 4)
        c_1 = 2 / ((self.dim + 1.3)**2 + mu)
        c_mu = min(1 - c_1, 2 * (mu - 1 + 1/mu) / ((self.dim + 2)**2 + 2*mu))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu - 1) / (self.dim + 1)) - 1) + c_sigma

        while self.eval_count < self.budget:
            # Sample population
            x = self.sample_population()
            
            # Clip individuals to respect boundaries
            lb = func.bounds.lb
            ub = func.bounds.ub
            x = np.clip(x, lb, ub)

            # Evaluate population
            fitness = np.array([func(xi) for xi in x])
            self.eval_count += self.popsize

            # Sort by fitness
            idx = np.argsort(fitness)
            fitness = fitness[idx]
            x = x[idx]

            # Update optimal solution
            if fitness[0] < self.f_opt:
                self.f_opt = fitness[0]
                self.x_opt = x[0]

            # Update mean
            xmean = np.mean(x[:mu], axis=0)
            self.ps = (1 - c_sigma) * self.ps + np.sqrt(c_sigma * (2 - c_sigma)) * (np.linalg.solve(np.linalg.cholesky(self.C), (xmean - self.mean) / self.sigma))
            
            hsig = (np.linalg.norm(self.ps) / np.sqrt(1 - (1 - c_sigma)**(2 * self.eval_count / self.popsize)) / self.chiN) < (1.4 + 2 / (self.dim + 1))
            self.pc = (1 - c_c) * self.pc + hsig * np.sqrt(c_c * (2 - c_c)) * (xmean - self.mean) / self.sigma
            self.mean = xmean

            # Update covariance matrix
            self.C = (1 - c_1 - c_mu) * self.C + c_1 * np.outer(self.pc, self.pc) + c_mu * sum(np.outer((x[i] - self.mean) / self.sigma, (x[i] - self.mean) / self.sigma) for i in range(mu))

            # Adapt step size
            gradient_norm = self.estimate_gradient_norm(func, self.mean)
            self.adapt_step_size(gradient_norm)

            if self.eval_count >= self.budget:
                break
                
        return self.f_opt, self.x_opt