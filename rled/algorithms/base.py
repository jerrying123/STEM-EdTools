"""
Base class for reinforcement learning algorithms.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

class RLAlgorithm(ABC):
    """Base class for all reinforcement learning algorithms."""
    
    def __init__(self, 
                 action_space_size: int,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.99,
                 epsilon: float = 0.1):
        """
        Initialize the RL algorithm.
        
        Args:
            action_space_size: Size of the action space
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Exploration rate for epsilon-greedy policy
        """
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # Initialize Q-table as a dictionary
        self.q_table: Dict[Any, Any] = {}
        
    @abstractmethod
    def select_action(self, state: Any) -> int:
        """
        Select an action using the current policy.
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        pass
    
    @abstractmethod
    def update(self, 
              state: Any,
              action: int,
              reward: float,
              next_state: Any,
              next_action: int = None) -> None:
        """
        Update the algorithm's parameters based on the observed transition.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            next_action: Next action (if applicable)
        """
        pass
    
    def get_policy(self) -> Dict[Any, int]:
        """
        Get the current policy (greedy with respect to Q-values).
        
        Returns:
            Dictionary mapping states to actions
        """
        return {state: max(actions, key=actions.get) for state, actions in self.q_table.items()}
    
    def get_state_value(self, state: Any) -> float:
        """
        Get the value of a state under the current policy.
        
        Args:
            state: State to evaluate
            
        Returns:
            Value of the state
        """
        if state not in self.q_table:
            return 0.0
        return max(self.q_table[state].values())
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get the current algorithm parameters.
        
        Returns:
            Dictionary of parameters
        """
        return {
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon
        }
    
    def ensure_state_exists(self, state: Any):
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in range(self.action_space_size)} 