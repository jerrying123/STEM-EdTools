"""
Base class for deep reinforcement learning algorithms using PyTorch.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
from collections import deque
import random

class DeepRLAlgorithm(ABC):
    """Base class for deep reinforcement learning algorithms."""
    
    def __init__(self, 
                 state_size: int,
                 action_size: int,
                 learning_rate: float = 0.001,
                 discount_factor: float = 0.99,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 hidden_size: int = 64,
                 device: str = None):
        """
        Initialize the deep RL algorithm.
        
        Args:
            state_size: Size of the state space
            action_size: Size of the action space
            learning_rate: Learning rate for neural network
            discount_factor: Discount factor (gamma)
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon value
            hidden_size: Size of hidden layers in neural network
            device: Device to run on ('cpu' or 'cuda')
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.hidden_size = hidden_size
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize networks (to be implemented by subclasses)
        self.q_network = None
        self.optimizer = None
        
        # Experience replay buffer
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
    def create_network(self):
        """Create the neural network architecture"""
        return nn.Sequential(
            nn.Linear(self.state_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.action_size)
        ).to(self.device)
    
    def select_action(self, state):
        """Select action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        
        # Convert state to tensor
        if isinstance(state, tuple):
            state = np.array(state)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get Q-values from network
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        if isinstance(state, tuple):
            state = np.array(state)
        if isinstance(next_state, tuple):
            next_state = np.array(next_state)
        
        self.memory.append((state, action, reward, next_state, done))
    
    def update_epsilon(self):
        """Decay epsilon for exploration"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    @abstractmethod
    def update(self, state, action, reward, next_state, done=False):
        """Update the algorithm's parameters"""
        pass
    
    @abstractmethod
    def replay(self):
        """Train the network using experience replay"""
        pass
    
    def get_parameters(self):
        """Get algorithm parameters"""
        return {
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'hidden_size': self.hidden_size,
            'device': str(self.device)
        }
    
    def get_policy(self):
        """Get current policy (for compatibility with tabular methods)"""
        # For deep RL, we can't enumerate all states, so return empty dict
        return {}
    
    def get_state_value(self, state):
        """Get value of a state"""
        if isinstance(state, tuple):
            state = np.array(state)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        return q_values.max().item()
    
    def save_model(self, filepath):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'parameters': self.get_parameters()
        }, filepath)
    
    def load_model(self, filepath):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon'] 