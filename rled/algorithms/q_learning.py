"""
Q-Learning algorithm implementation.
"""
import numpy as np
from .base import RLAlgorithm

class QLearning(RLAlgorithm):
    """Q-Learning algorithm implementation."""
    
    def select_action(self, state):
        self.ensure_state_exists(state)
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_space_size)
        else:
            return max(self.q_table[state], key=self.q_table[state].get)
    
    def update(self, state, action, reward, next_state, next_action=None):
        self.ensure_state_exists(state)
        self.ensure_state_exists(next_state)
        current_q = self.q_table[state][action]
        best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
        next_q = self.q_table[next_state][best_next_action]
        self.q_table[state][action] = current_q + self.learning_rate * (
            reward + self.discount_factor * next_q - current_q
        ) 