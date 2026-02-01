import numpy as np

class SimplifiedCMAES:
    def __init__(self, budget=10000, dim=10, sigma0=0.5, step_size_factor=0.9):
        self.budget = budget
        self.dim = dim
        self.sigma0 = sigma0
        self.lb = -5.0
        self.ub = 5.0
        self.step_size_factor = step_size_factor

    def __call__(self, func):
        # Initialize variables
        mu = np.random.uniform(self.lb, self.ub, size=self.dim)
        sigma = self.sigma0  # Overall standard deviation
        f_opt = np.Inf
        x_opt = None
        eval_count = 0

        while eval_count < self.budget:
            # Generate a single candidate solution
            z = np.random.normal(0, 1, size=self.dim)
            x = mu + sigma * z

            # Repair individual outside the bounds
            x = np.clip(x, self.lb, self.ub)

            # Evaluate candidate
            f = func(x)
            eval_count += 1

            # Update the mean and step size
            if f < f_opt:
                f_opt = f
                x_opt = x.copy()
                mu = x.copy() # Move mean towards the better solution
                sigma *= self.step_size_factor  # Reduce step size

            else:
                sigma /= self.step_size_factor # increase step size

            sigma = np.clip(sigma, 1e-6, 1)  # Keep sigma within reasonable bounds

        return f_opt, x_opt