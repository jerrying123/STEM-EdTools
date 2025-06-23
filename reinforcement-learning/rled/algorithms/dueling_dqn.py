"""
Dueling Deep Q-Network (Dueling DQN) algorithm implementation.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from .deep_base import DeepRLAlgorithm

class DuelingNetwork(nn.Module):
    """Dueling network architecture that separates value and advantage streams"""
    
    def __init__(self, state_size, action_size, hidden_size=64):
        super(DuelingNetwork, self).__init__()
        
        # Shared feature layers
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )
    
    def forward(self, x):
        # Extract features
        features = self.feature_layer(x)
        
        # Compute value and advantage
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage using dueling architecture
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values

class DuelingDQN(DeepRLAlgorithm):
    """Dueling Deep Q-Network algorithm implementation."""
    
    def __init__(self, state_size, action_size, learning_rate=0.001, 
                 discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995, 
                 epsilon_min=0.01, hidden_size=64, device=None, 
                 target_update_frequency=100):
        super().__init__(state_size, action_size, learning_rate, discount_factor,
                        epsilon, epsilon_decay, epsilon_min, hidden_size, device)
        
        # Create main and target dueling networks
        self.q_network = DuelingNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_network = DuelingNetwork(state_size, action_size, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Training frequency
        self.update_frequency = 4
        self.target_update_frequency = target_update_frequency
        self.step_count = 0
    
    def update(self, state, action, reward, next_state, done=False):
        """Store experience and train if enough samples available"""
        # Store experience in replay buffer
        self.remember(state, action, reward, next_state, done)
        
        # Train the network every update_frequency steps
        self.step_count += 1
        if self.step_count % self.update_frequency == 0:
            if len(self.memory) > self.batch_size:
                self.replay()
        
        # Update target network
        if self.step_count % self.target_update_frequency == 0:
            self.update_target_network()
        
        # Decay epsilon
        self.update_epsilon()
    
    def replay(self):
        """Train the network using experience replay"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample random batch from memory
        batch = random.sample(self.memory, self.batch_size)
        
        # Convert to numpy arrays first, then to tensors (more efficient)
        states = torch.FloatTensor(np.array([e[0] for e in batch])).to(self.device)
        actions = torch.LongTensor(np.array([e[1] for e in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([e[2] for e in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([e[3] for e in batch])).to(self.device)
        dones = torch.BoolTensor(np.array([e[4] for e in batch])).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values using target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.discount_factor * next_q_values * ~dones)
        
        # Compute loss
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def get_q_values(self, state):
        """Get Q-values for a given state"""
        if isinstance(state, tuple):
            state = np.array(state)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        return q_values.cpu().numpy().flatten()
    
    def get_value_and_advantage(self, state):
        """Get separate value and advantage estimates (for educational purposes)"""
        if isinstance(state, tuple):
            state = np.array(state)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.q_network.feature_layer(state_tensor)
            value = self.q_network.value_stream(features)
            advantage = self.q_network.advantage_stream(features)
        
        return value.cpu().numpy().flatten()[0], advantage.cpu().numpy().flatten()
    
    def get_parameters(self):
        """Get algorithm parameters"""
        params = super().get_parameters()
        params['target_update_frequency'] = self.target_update_frequency
        return params 