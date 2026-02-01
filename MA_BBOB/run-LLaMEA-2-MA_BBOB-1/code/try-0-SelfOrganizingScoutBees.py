import numpy as np

class SelfOrganizingScoutBees:
    def __init__(self, budget=10000, dim=10, n_scouts=10, scout_memory=5, initial_step_size=0.5):
        self.budget = budget
        self.dim = dim
        self.n_scouts = n_scouts
        self.scout_memory = scout_memory
        self.step_size = initial_step_size
        self.lb = -5.0
        self.ub = 5.0
        self.scouts_pos = np.random.uniform(self.lb, self.ub, size=(self.n_scouts, self.dim))
        self.scouts_fitness = np.full(self.n_scouts, np.inf)
        self.scouts_memory_pos = np.zeros((self.n_scouts, self.scout_memory, self.dim))
        self.scouts_memory_fitness = np.full((self.n_scouts, self.scout_memory), np.inf)
        self.f_opt = np.inf
        self.x_opt = None
        self.eval_count = 0

    def scout_move(self, scout_id, func):
        # Adaptive step size adjustment
        explore_prob = 0.2
        if np.random.rand() < explore_prob:
            # Exploration: Random move with adaptive step size
            new_pos = self.scouts_pos[scout_id] + np.random.uniform(-self.step_size, self.step_size, self.dim)
        else:
            # Exploitation: Move towards best memory
            best_memory_idx = np.argmin(self.scouts_memory_fitness[scout_id])
            new_pos = self.scouts_pos[scout_id] + np.random.uniform(-self.step_size, self.step_size, self.dim) * (self.scouts_memory_pos[scout_id, best_memory_idx] - self.scouts_pos[scout_id])

        new_pos = np.clip(new_pos, self.lb, self.ub)
        new_fitness = func(new_pos)
        self.eval_count += 1

        # Update scout's position and fitness
        if new_fitness < self.scouts_fitness[scout_id]:
            self.scouts_pos[scout_id] = new_pos
            self.scouts_fitness[scout_id] = new_fitness

            # Update scout's memory
            self.scouts_memory_pos[scout_id, :-1] = self.scouts_memory_pos[scout_id, 1:]
            self.scouts_memory_fitness[scout_id, :-1] = self.scouts_memory_fitness[scout_id, 1:]
            self.scouts_memory_pos[scout_id, -1] = new_pos
            self.scouts_memory_fitness[scout_id, -1] = new_fitness
        
        return new_fitness, new_pos

    def adapt_step_size(self):
      # Adjust step size based on success rate (simplified)
      success_rate = np.sum(self.scouts_fitness < np.inf) / self.n_scouts
      if success_rate > 0.5:
          self.step_size *= 1.1  # Increase step size if successful
      else:
          self.step_size *= 0.9  # Decrease step size if unsuccessful

      self.step_size = np.clip(self.step_size, 0.01, 1.0)  # Limit step size

    def __call__(self, func):
        # Initialize scouts and their memory
        for i in range(self.n_scouts):
            self.scouts_fitness[i] = func(self.scouts_pos[i])
            self.eval_count += 1
            self.scouts_memory_pos[i, -1] = self.scouts_pos[i]
            self.scouts_memory_fitness[i, -1] = self.scouts_fitness[i]

            if self.scouts_fitness[i] < self.f_opt:
                self.f_opt = self.scouts_fitness[i]
                self.x_opt = self.scouts_pos[i]

        while self.eval_count < self.budget:
            for i in range(self.n_scouts):
                f, x = self.scout_move(i, func)

                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = x

            self.adapt_step_size()
            
        return self.f_opt, self.x_opt