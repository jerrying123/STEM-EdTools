"""
SARSA (State-Action-Reward-State-Action) algorithm implementation.
"""
import numpy as np
from .base import RLAlgorithm

class SARSA(RLAlgorithm):
    """SARSA algorithm implementation."""
    
    def select_action(self, state):
        self.ensure_state_exists(state)
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_space_size)
        else:
            return max(self.q_table[state], key=self.q_table[state].get)
    
    def update(self, state, action, reward, next_state, next_action):
        self.ensure_state_exists(state)
        self.ensure_state_exists(next_state)
        if next_action is None:
            raise ValueError("SARSA requires next_action for update")
        current_q = self.q_table[state][action]
        next_q = self.q_table[next_state][next_action]
        self.q_table[state][action] = current_q + self.learning_rate * (
            reward + self.discount_factor * next_q - current_q
        ) 