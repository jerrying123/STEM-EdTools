"""
Expected SARSA algorithm implementation.
"""
import numpy as np
from .base import RLAlgorithm

class ExpectedSARSA(RLAlgorithm):
    """Expected SARSA algorithm implementation."""
    
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
        
        # Calculate expected value of next state under current policy
        expected_q = 0.0
        best_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
        
        for a in range(self.action_space_size):
            if a == best_action:
                # Probability of selecting best action (greedy + exploration)
                prob = 1 - self.epsilon + (self.epsilon / self.action_space_size)
            else:
                # Probability of selecting non-best action (exploration only)
                prob = self.epsilon / self.action_space_size
            
            expected_q += prob * self.q_table[next_state][a]
        
        # Update Q-value using expected value
        self.q_table[state][action] = current_q + self.learning_rate * (
            reward + self.discount_factor * expected_q - current_q
        ) 