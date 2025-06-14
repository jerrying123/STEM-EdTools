"""
REINFORCE (Policy Gradient) algorithm implementation.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from .deep_base import DeepRLAlgorithm

class PolicyNetwork(nn.Module):
    """Policy network that outputs action probabilities"""
    
    def __init__(self, state_size, action_size, hidden_size=64):
        super(PolicyNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.network(x)

class REINFORCE(DeepRLAlgorithm):
    """REINFORCE (Policy Gradient) algorithm implementation."""
    
    def __init__(self, state_size, action_size, learning_rate=0.001, 
                 discount_factor=0.99, hidden_size=64, device=None):
        # REINFORCE doesn't use epsilon-greedy, so set epsilon to 0
        super().__init__(state_size, action_size, learning_rate, discount_factor,
                        epsilon=0.0, epsilon_decay=1.0, epsilon_min=0.0, 
                        hidden_size=hidden_size, device=device)
        
        # Create policy network (not Q-network)
        self.policy_network = PolicyNetwork(state_size, action_size, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        
        # Store episode data for batch updates
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_log_probs = []
        
        # For compatibility with base class
        self.q_network = self.policy_network
    
    def select_action(self, state):
        """Select action using policy network"""
        if isinstance(state, tuple):
            state = np.array(state)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get action probabilities
        action_probs = self.policy_network(state_tensor)
        
        # Sample action from probability distribution
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        
        # Store log probability for training
        log_prob = action_dist.log_prob(action)
        self.episode_log_probs.append(log_prob)
        
        return action.item()
    
    def start_episode(self):
        """Start a new episode - clear episode data"""
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_log_probs = []
    
    def update(self, state, action, reward, next_state, done=False):
        """Store transition data for end-of-episode processing"""
        if isinstance(state, tuple):
            state = np.array(state)
        
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
    
    def end_episode(self):
        """Process complete episode and update policy"""
        if not self.episode_rewards:
            return
        
        # Calculate discounted returns
        returns = []
        G = 0
        for reward in reversed(self.episode_rewards):
            G = reward + self.discount_factor * G
            returns.insert(0, G)
        
        # Convert to tensor
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize returns for stability
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate policy loss
        policy_loss = []
        for log_prob, G in zip(self.episode_log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        policy_loss = torch.stack(policy_loss).sum()
        
        # Update policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Clear episode data
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_log_probs = []
        
        return policy_loss.item()
    
    def get_action_probabilities(self, state):
        """Get action probabilities for a given state"""
        if isinstance(state, tuple):
            state = np.array(state)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs = self.policy_network(state_tensor)
        
        return action_probs.cpu().numpy().flatten()
    
    def get_state_value(self, state):
        """Get estimated value of a state (using max action probability as proxy)"""
        action_probs = self.get_action_probabilities(state)
        return np.max(action_probs)
    
    def replay(self):
        """REINFORCE doesn't use experience replay"""
        pass
    
    def remember(self, state, action, reward, next_state, done):
        """REINFORCE doesn't use experience replay buffer"""
        pass 