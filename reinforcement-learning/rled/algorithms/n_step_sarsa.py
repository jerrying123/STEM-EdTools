"""
n-step SARSA algorithm implementation.
"""
import numpy as np
from collections import deque
from .base import RLAlgorithm

class NStepSARSA(RLAlgorithm):
    """n-step SARSA algorithm implementation."""
    
    def __init__(self, action_space_size, learning_rate=0.1, discount_factor=0.99, epsilon=0.1, n_steps=3):
        super().__init__(action_space_size, learning_rate, discount_factor, epsilon)
        self.n_steps = n_steps
        # Store recent transitions for n-step updates
        self.states = deque(maxlen=n_steps + 1)
        self.actions = deque(maxlen=n_steps + 1)
        self.rewards = deque(maxlen=n_steps)
        self.step_count = 0
    
    def select_action(self, state):
        self.ensure_state_exists(state)
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_space_size)
        else:
            return max(self.q_table[state], key=self.q_table[state].get)
    
    def start_episode(self):
        """Start a new episode - clear buffers"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.step_count = 0
    
    def update(self, state, action, reward, next_state, next_action=None):
        """Update using n-step SARSA"""
        self.ensure_state_exists(state)
        self.ensure_state_exists(next_state)
        
        # Store current transition
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.step_count += 1
        
        # If we have enough steps, perform n-step update
        if len(self.states) > self.n_steps:
            self._perform_n_step_update()
    
    def _perform_n_step_update(self):
        """Perform n-step SARSA update"""
        if len(self.states) < self.n_steps + 1:
            return
        
        # Get the state and action to update (n steps ago)
        update_state = self.states[0]
        update_action = self.actions[0]
        
        # Calculate n-step return
        G = 0.0
        for i in range(self.n_steps):
            G += (self.discount_factor ** i) * self.rewards[i]
        
        # Add discounted value of state n steps ahead
        if len(self.states) == self.n_steps + 1:
            future_state = self.states[-1]
            if len(self.actions) == self.n_steps + 1:
                future_action = self.actions[-1]
                G += (self.discount_factor ** self.n_steps) * self.q_table[future_state][future_action]
        
        # Update Q-value
        current_q = self.q_table[update_state][update_action]
        self.q_table[update_state][update_action] = current_q + self.learning_rate * (G - current_q)
    
    def end_episode(self, final_state=None):
        """Process remaining updates at episode end"""
        # Add final state if provided
        if final_state is not None:
            self.states.append(final_state)
        
        # Perform remaining updates
        while len(self.states) > 1 and len(self.rewards) > 0:
            self._perform_n_step_update()
            # Remove oldest entries to process next update
            if len(self.states) > 0:
                self.states.popleft()
            if len(self.actions) > 0:
                self.actions.popleft()
            if len(self.rewards) > 0:
                self.rewards.popleft()
    
    def get_parameters(self):
        """Get algorithm parameters including n_steps"""
        params = super().get_parameters()
        params['n_steps'] = self.n_steps
        return params 