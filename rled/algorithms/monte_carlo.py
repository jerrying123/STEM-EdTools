"""
Monte Carlo Control algorithm implementation.
"""
import numpy as np
from .base import RLAlgorithm

class MonteCarloControl(RLAlgorithm):
    """Monte Carlo Control algorithm implementation."""
    
    def __init__(self, action_space_size, learning_rate=0.1, discount_factor=0.99, epsilon=0.1):
        super().__init__(action_space_size, learning_rate, discount_factor, epsilon)
        # Track returns for each state-action pair
        self.returns = {}
        # Store episode data for batch updates
        self.episode_data = []
    
    def ensure_state_exists(self, state):
        """Ensure state exists in Q-table and returns tracking"""
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in range(self.action_space_size)}
        if state not in self.returns:
            self.returns[state] = {a: [] for a in range(self.action_space_size)}
    
    def select_action(self, state):
        self.ensure_state_exists(state)
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_space_size)
        else:
            return max(self.q_table[state], key=self.q_table[state].get)
    
    def start_episode(self):
        """Start a new episode - clear episode data"""
        self.episode_data = []
    
    def store_transition(self, state, action, reward):
        """Store transition data for end-of-episode processing"""
        self.episode_data.append((state, action, reward))
    
    def update(self, state, action, reward, next_state, next_action=None):
        """Store transition - actual update happens at episode end"""
        self.store_transition(state, action, reward)
    
    def end_episode(self):
        """Process complete episode and update Q-values"""
        if not self.episode_data:
            return
        
        # Calculate returns for each state-action pair in the episode
        G = 0  # Return
        visited_state_actions = set()
        
        # Process episode backwards to calculate returns
        for i in reversed(range(len(self.episode_data))):
            state, action, reward = self.episode_data[i]
            G = self.discount_factor * G + reward
            
            # First-visit Monte Carlo: only update if this is the first visit to (s,a)
            state_action = (state, action)
            if state_action not in visited_state_actions:
                visited_state_actions.add(state_action)
                
                self.ensure_state_exists(state)
                
                # Store return
                self.returns[state][action].append(G)
                
                # Update Q-value as average of returns
                self.q_table[state][action] = np.mean(self.returns[state][action])
        
        # Clear episode data
        self.episode_data = [] 