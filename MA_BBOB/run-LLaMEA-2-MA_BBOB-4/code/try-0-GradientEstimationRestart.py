import numpy as np

class GradientEstimationRestart:
    def __init__(self, budget=10000, dim=10, step_size=0.1, num_samples=10, restart_patience=1000):
        self.budget = budget
        self.dim = dim
        self.step_size = step_size
        self.num_samples = num_samples
        self.restart_patience = restart_patience
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        x = np.random.uniform(self.lb, self.ub, size=self.dim)
        f_current = func(x)
        eval_count = 1
        f_opt = f_current
        x_opt = x.copy()
        no_improvement_count = 0

        while eval_count < self.budget:
            # Estimate gradient
            gradient = np.zeros(self.dim)
            for _ in range(self.num_samples):
                direction = np.random.normal(0, 1, size=self.dim)
                direction /= np.linalg.norm(direction)  # Normalize
                x_perturbed = x + self.step_size * direction
                x_perturbed = np.clip(x_perturbed, self.lb, self.ub)
                f_perturbed = func(x_perturbed)
                eval_count += 1
                gradient += (f_perturbed - f_current) * direction

                if eval_count >= self.budget:
                    break

            gradient /= self.num_samples * self.step_size

            # Update position
            x_new = x - self.step_size * gradient
            x_new = np.clip(x_new, self.lb, self.ub)
            f_new = func(x_new)
            eval_count += 1

            if f_new < f_current:
                x = x_new
                f_current = f_new
                no_improvement_count = 0
                if f_new < f_opt:
                    f_opt = f_new
                    x_opt = x.copy()
            else:
                no_improvement_count += 1
                self.step_size *= 0.9  # Reduce step size if no improvement

            # Restart if no improvement for too long
            if no_improvement_count > self.restart_patience:
                x = np.random.uniform(self.lb, self.ub, size=self.dim)
                f_current = func(x)
                eval_count += 1
                self.step_size = 0.1 # Reinitialize stepsize
                no_improvement_count = 0


            if eval_count >= self.budget:
                break

        return f_opt, x_opt