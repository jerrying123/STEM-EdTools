"""
Algorithms module for RLEd.
Provides implementations of reinforcement learning algorithms.
"""

# Tabular RL algorithms
from .q_learning import QLearning
from .sarsa import SARSA
from .expected_sarsa import ExpectedSARSA
from .double_q_learning import DoubleQLearning
from .monte_carlo import MonteCarloControl
from .n_step_sarsa import NStepSARSA

# Deep RL algorithms
from .dqn import DQN
from .double_dqn import DoubleDQN
from .dueling_dqn import DuelingDQN
from .reinforce import REINFORCE

__all__ = [
    # Tabular methods
    'QLearning', 
    'SARSA', 
    'ExpectedSARSA', 
    'DoubleQLearning', 
    'MonteCarloControl', 
    'NStepSARSA',
    # Deep methods
    'DQN',
    'DoubleDQN',
    'DuelingDQN',
    'REINFORCE'
] 