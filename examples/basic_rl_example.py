"""
Example script demonstrating the use of basic RL algorithms.
"""
import gymnasium as gym
import numpy as np
from rled.algorithms import QLearning, SARSA

def discretize_state(observation, bins):
    """Discretize continuous state space into bins."""
    return tuple(np.digitize(observation[i], bins[i]) for i in range(len(observation)))

def create_state_bins(env, num_bins=10):
    """Create bins for discretizing the state space."""
    state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
    bins = [np.linspace(low, high, num_bins) for low, high in state_bounds]
    return bins

def main():
    print("Starting RL example script...")
    
    # Create environment
    print("Creating CartPole environment...")
    env = gym.make('CartPole-v1')
    state_bins = create_state_bins(env)
    print(f"Environment created. State space dimensions: {len(state_bins)}")
    
    # Initialize algorithms
    action_size = env.action_space.n
    print(f"Action space size: {action_size}")
    
    print("Initializing Q-Learning algorithm...")
    q_learning = QLearning(
        action_space_size=action_size,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=0.1
    )
    
    print("Initializing SARSA algorithm...")
    sarsa = SARSA(
        action_space_size=action_size,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=0.1
    )
    
    # Training parameters
    episodes = 100  # Reduced for debugging
    max_steps = 500
    print(f"Training parameters: {episodes} episodes, max {max_steps} steps per episode")
    
    # Train Q-Learning
    print("\nTraining Q-Learning...")
    for episode in range(episodes):
        state, _ = env.reset()
        state = discretize_state(state, state_bins)
        total_reward = 0
        
        if episode == 0:
            print(f"First episode - Initial state: {state}")
        
        for step in range(max_steps):
            action = q_learning.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = discretize_state(next_state, state_bins)
            
            q_learning.update(state, action, reward, next_state)
            
            state = next_state
            total_reward += reward
            
            if terminated or truncated:
                break
                
        if (episode + 1) % 10 == 0:  # Print every 10 episodes for debugging
            print(f"Q-Learning Episode {episode + 1}, Total Reward: {total_reward}")
    
    print(f"Q-Learning training completed. Q-table size: {len(q_learning.q_table)}")
    
    # Train SARSA
    print("\nTraining SARSA...")
    for episode in range(episodes):
        state, _ = env.reset()
        state = discretize_state(state, state_bins)
        action = sarsa.select_action(state)
        total_reward = 0
        
        if episode == 0:
            print(f"First episode - Initial state: {state}, Initial action: {action}")
        
        for step in range(max_steps):
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = discretize_state(next_state, state_bins)
            next_action = sarsa.select_action(next_state)
            
            sarsa.update(state, action, reward, next_state, next_action)
            
            state = next_state
            action = next_action
            total_reward += reward
            
            if terminated or truncated:
                break
                
        if (episode + 1) % 10 == 0:  # Print every 10 episodes for debugging
            print(f"SARSA Episode {episode + 1}, Total Reward: {total_reward}")
    
    print(f"SARSA training completed. Q-table size: {len(sarsa.q_table)}")
    
    env.close()
    print("Script completed successfully!")
        
if __name__ == "__main__":
    main() 