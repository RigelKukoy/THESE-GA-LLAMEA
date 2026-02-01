import numpy as np

class AdaptiveStepSizeRandomSearch:
    def __init__(self, budget=10000, dim=10, initial_step_size=0.1):
        self.budget = budget
        self.dim = dim
        self.step_size = initial_step_size
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        x = np.random.uniform(self.lb, self.ub, size=self.dim)
        f_best = func(x)
        eval_count = 1
        x_best = x.copy()

        success_count = 0
        total_trials = 0

        while eval_count < self.budget:
            # Generate a new candidate solution by adding a random step
            x_new = x + np.random.normal(0, self.step_size, size=self.dim)

            # Clip the solution to stay within bounds
            x_new = np.clip(x_new, self.lb, self.ub)

            f_new = func(x_new)
            eval_count += 1

            if f_new < f_best:
                f_best = f_new
                x_best = x_new.copy()
                x = x_new
                success_count += 1
            else:
                # Keep the old solution
                pass
            
            total_trials += 1

            # Adjust step size based on success rate
            if total_trials % 100 == 0:
                success_rate = success_count / total_trials
                if success_rate > 0.2:
                    self.step_size *= 1.1  # Increase step size
                elif success_rate < 0.1:
                    self.step_size *= 0.9  # Decrease step size

                # Reset counters
                success_count = 0
                total_trials = 0
                
            if eval_count >= self.budget:
                 break

        return f_best, x_best