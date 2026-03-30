import gymnasium as gym
import numpy as np
from gymnasium import spaces


class GridEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Observation: preparedness, time_since_maint, budget
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(2)

        # ---- Dynamics ----
        self.decay_rate = 0.02
        self.maintenance_boost = 0.1

        # ---- Tail Risk ----
        self.base_failure = 0.0003
        self.k = 5
        self.max_failure_prob = 0.15
        self.catastrophe_scale = 4000

        # ---- Budget ----
        self.initial_budget = 120
        self.maintenance_cost = 5
        self.passive_budget_drain = 0.3

        self.max_steps = 200
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.preparedness = 1.0
        self.time_since_maintenance = 0
        self.budget = self.initial_budget
        self.step_count = 0

        return self._get_state(), {}

    def _get_state(self):
        return np.array([
            self.preparedness,
            min(self.time_since_maintenance / 50.0, 1.0),
            min(self.budget / self.initial_budget, 1.0)
        ], dtype=np.float32)
    
    def step(self, action):
        reward = 0
        self.step_count += 1

        # 1. Update Dynamics
        self.budget -= self.passive_budget_drain
        self.preparedness -= self.decay_rate

        # 2. Handle Action
        if action == 1 and self.budget >= self.maintenance_cost:
            self.budget -= self.maintenance_cost
            self.preparedness += self.maintenance_boost
            self.time_since_maintenance = 0
            reward -= 1 
        else:
            self.time_since_maintenance += 1

        self.preparedness = np.clip(self.preparedness, 0, 1)

        # 3. Calculate Failure Probability
        p_failure = self.base_failure * np.exp(self.k * (1 - self.preparedness))
        if self.budget <= 0:
            p_failure *= 2
        p_failure = min(p_failure, self.max_failure_prob)

        # 4. Determine if failure happened
        failure = np.random.rand() < p_failure

        # --- RISK AWARE PENALTIES START HERE ---
        
        # Mild continuous risk awareness (Linear penalty)
        risk_penalty = 20 * (1 - self.preparedness)
        reward -= risk_penalty

        # Keep original catastrophe (NO extra -2000)
        if failure:
            severity = self.catastrophe_scale * (1 - self.preparedness)
            reward -= severity
            
        # --- RISK AWARE PENALTIES END HERE ---

        terminated = False
        truncated = self.step_count >= self.max_steps
        info = {"failure": failure, "p_failure": p_failure}

        return self._get_state(), reward, terminated, truncated, info
