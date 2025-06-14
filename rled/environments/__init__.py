"""
Environments module for RLEd.
Provides wrappers and utilities for working with Gymnasium environments.
"""

from .gridworld_treasure import GridWorldTreasureEnv, register_gridworld_treasure

# Register custom environments
register_gridworld_treasure()

__all__ = ['GridWorldTreasureEnv', 'register_gridworld_treasure'] 