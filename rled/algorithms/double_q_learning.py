"""
Double Q-Learning algorithm implementation.
"""
import numpy as np
from .base import RLAlgorithm

class DoubleQLearning(RLAlgorithm):
    """Double Q-Learning algorithm implementation."""
    
    def __init__(self, action_space_size, learning_rate=0.1, discount_factor=0.99, epsilon=0.1):
        super().__init__(action_space_size, learning_rate, discount_factor, epsilon)
        # Initialize two Q-tables
        self.q_table_a = {}
        self.q_table_b = {}
        # Keep the main q_table for compatibility (average of both)
        self.q_table = {}
    
    def ensure_state_exists(self, state):
        """Ensure state exists in both Q-tables"""
        if state not in self.q_table_a:
            self.q_table_a[state] = {a: 0.0 for a in range(self.action_space_size)}
        if state not in self.q_table_b:
            self.q_table_b[state] = {a: 0.0 for a in range(self.action_space_size)}
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in range(self.action_space_size)}
    
    def select_action(self, state):
        self.ensure_state_exists(state)
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_space_size)
        else:
            # Use average of both Q-tables for action selection
            avg_q_values = {}
            for a in range(self.action_space_size):
                avg_q_values[a] = (self.q_table_a[state][a] + self.q_table_b[state][a]) / 2
            return max(avg_q_values, key=avg_q_values.get)
    
    def update(self, state, action, reward, next_state, next_action=None):
        self.ensure_state_exists(state)
        self.ensure_state_exists(next_state)
        
        # Randomly choose which Q-table to update
        if np.random.random() < 0.5:
            # Update Q_A using Q_B for next state value
            best_next_action_a = max(self.q_table_a[next_state], key=self.q_table_a[next_state].get)
            next_q = self.q_table_b[next_state][best_next_action_a]
            current_q = self.q_table_a[state][action]
            self.q_table_a[state][action] = current_q + self.learning_rate * (
                reward + self.discount_factor * next_q - current_q
            )
        else:
            # Update Q_B using Q_A for next state value
            best_next_action_b = max(self.q_table_b[next_state], key=self.q_table_b[next_state].get)
            next_q = self.q_table_a[next_state][best_next_action_b]
            current_q = self.q_table_b[state][action]
            self.q_table_b[state][action] = current_q + self.learning_rate * (
                reward + self.discount_factor * next_q - current_q
            )
        
        # Update the main Q-table with average values for compatibility
        for a in range(self.action_space_size):
            self.q_table[state][a] = (self.q_table_a[state][a] + self.q_table_b[state][a]) / 2 