#!/usr/bin/env python3
"""
GridWorld Treasure Hunt Example

This example demonstrates the custom GridWorld Treasure Hunt environment
with both tabular and deep reinforcement learning algorithms.

The environment features:
- Grid-based world with treasures, obstacles, and energy management
- Multi-objective rewards (treasure collection, energy efficiency, time)
- 5 actions: Up, Right, Down, Left, Stay (to recover energy)
- Rich observation space including position, energy, and grid states
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from rled.algorithms import QLearning, DQN
from rled.environments import GridWorldTreasureEnv

def run_tabular_example():
    """Run Q-Learning on GridWorld Treasure Hunt"""
    print("=" * 60)
    print("GridWorld Treasure Hunt - Q-Learning Example")
    print("=" * 60)
    
    # Create environment
    env = GridWorldTreasureEnv(
        grid_size=6,
        num_treasures=3,
        num_obstacles=4,
        max_energy=80,
        max_steps=150
    )
    
    print(f"Environment: {env.grid_size}x{env.grid_size} grid")
    print(f"Treasures: {env.num_treasures}, Obstacles: {env.num_obstacles}")
    print(f"Max Energy: {env.max_energy}, Max Steps: {env.max_steps}")
    print(f"Observation Space: {env.observation_space.shape}")
    print(f"Action Space: {env.action_space}")
    
    # Initialize Q-Learning agent
    # For tabular methods, we'll use a simplified state representation
    agent = QLearning(
        action_space_size=env.action_space.n,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.3
    )
    
    # Training parameters
    episodes = 200
    rewards_history = []
    
    print(f"\nTraining Q-Learning for {episodes} episodes...")
    
    for episode in range(episodes):
        obs, info = env.reset()
        
        # Simplify state for tabular method (position + energy level)
        state = (int(obs[0]), int(obs[1]), int(obs[2] // 10))  # Discretize energy
        
        total_reward = 0
        
        for step in range(env.max_steps):
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            
            # Simplify next state
            next_state = (int(next_obs[0]), int(next_obs[1]), int(next_obs[2] // 10))
            
            agent.update(state, action, reward, next_state)
            
            state = next_state
            total_reward += reward
            
            if terminated or truncated:
                break
        
        rewards_history.append(total_reward)
        
        # Decay epsilon
        agent.epsilon = max(0.01, agent.epsilon * 0.995)
        
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:])
            print(f"Episode {episode + 1}: Average Reward = {avg_reward:.2f}, "
                  f"Epsilon = {agent.epsilon:.3f}")
    
    # Test the trained agent
    print("\nTesting trained Q-Learning agent...")
    test_rewards = []
    
    for test_episode in range(10):
        obs, info = env.reset()
        state = (int(obs[0]), int(obs[1]), int(obs[2] // 10))
        
        total_reward = 0
        agent.epsilon = 0.0  # Greedy policy
        
        for step in range(env.max_steps):
            action = agent.select_action(state)
            obs, reward, terminated, truncated, info = env.step(action)
            state = (int(obs[0]), int(obs[1]), int(obs[2] // 10))
            total_reward += reward
            
            if terminated or truncated:
                break
        
        test_rewards.append(total_reward)
        print(f"Test Episode {test_episode + 1}: Reward = {total_reward:.1f}, "
              f"Treasures = {info['treasures_collected']}/{info['total_treasures']}, "
              f"Steps = {info['steps_taken']}")
    
    print(f"Average Test Reward: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    
    # Visualize final state
    fig = env.render('matplotlib')
    plt.title("Final State - Q-Learning Agent")
    plt.show()
    
    env.close()
    return rewards_history

def run_deep_rl_example():
    """Run DQN on GridWorld Treasure Hunt"""
    print("\n" + "=" * 60)
    print("GridWorld Treasure Hunt - DQN Example")
    print("=" * 60)
    
    # Create environment
    env = GridWorldTreasureEnv(
        grid_size=6,
        num_treasures=3,
        num_obstacles=4,
        max_energy=80,
        max_steps=150
    )
    
    # Initialize DQN agent
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQN(
        state_size=state_size,
        action_size=action_size,
        learning_rate=0.001,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        hidden_size=128
    )
    
    print(f"DQN Agent: State Size = {state_size}, Action Size = {action_size}")
    print(f"Hidden Layer Size: {agent.hidden_size}")
    
    # Training parameters
    episodes = 300
    rewards_history = []
    
    print(f"\nTraining DQN for {episodes} episodes...")
    
    for episode in range(episodes):
        obs, info = env.reset()
        total_reward = 0
        
        for step in range(env.max_steps):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            
            # Update agent (DQN uses the full observation space)
            agent.update(obs, action, reward, next_obs, terminated or truncated)
            
            obs = next_obs
            total_reward += reward
            
            if terminated or truncated:
                break
        
        rewards_history.append(total_reward)
        
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:])
            print(f"Episode {episode + 1}: Average Reward = {avg_reward:.2f}, "
                  f"Epsilon = {agent.epsilon:.3f}")
    
    # Test the trained agent
    print("\nTesting trained DQN agent...")
    test_rewards = []
    
    for test_episode in range(10):
        obs, info = env.reset()
        total_reward = 0
        
        # Set epsilon to 0 for testing (greedy policy)
        original_epsilon = agent.epsilon
        agent.epsilon = 0.0
        
        for step in range(env.max_steps):
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        agent.epsilon = original_epsilon  # Restore epsilon
        
        test_rewards.append(total_reward)
        print(f"Test Episode {test_episode + 1}: Reward = {total_reward:.1f}, "
              f"Treasures = {info['treasures_collected']}/{info['total_treasures']}, "
              f"Steps = {info['steps_taken']}")
    
    print(f"Average Test Reward: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    
    # Visualize final state
    fig = env.render('matplotlib')
    plt.title("Final State - DQN Agent")
    plt.show()
    
    env.close()
    return rewards_history

def compare_algorithms():
    """Compare Q-Learning and DQN performance"""
    print("\n" + "=" * 60)
    print("Algorithm Comparison")
    print("=" * 60)
    
    # Run both algorithms
    print("Running Q-Learning...")
    qlearning_rewards = run_tabular_example()
    
    print("\nRunning DQN...")
    dqn_rewards = run_deep_rl_example()
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    
    # Moving averages for smoother plots
    window = 20
    
    def moving_average(data, window):
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    qlearning_ma = moving_average(qlearning_rewards, window)
    dqn_ma = moving_average(dqn_rewards, window)
    
    plt.subplot(1, 2, 1)
    plt.plot(qlearning_rewards, alpha=0.3, color='blue', label='Q-Learning (raw)')
    plt.plot(range(window-1, len(qlearning_rewards)), qlearning_ma, color='blue', label='Q-Learning (MA)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Q-Learning Performance')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(dqn_rewards, alpha=0.3, color='red', label='DQN (raw)')
    plt.plot(range(window-1, len(dqn_rewards)), dqn_ma, color='red', label='DQN (MA)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('DQN Performance')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print final comparison
    print("\nFinal Performance Comparison:")
    print(f"Q-Learning - Last 50 episodes average: {np.mean(qlearning_rewards[-50:]):.2f}")
    print(f"DQN - Last 50 episodes average: {np.mean(dqn_rewards[-50:]):.2f}")

def demonstrate_environment():
    """Demonstrate the environment features"""
    print("=" * 60)
    print("GridWorld Treasure Hunt Environment Demonstration")
    print("=" * 60)
    
    # Create a small environment for demonstration
    env = GridWorldTreasureEnv(
        grid_size=5,
        num_treasures=2,
        num_obstacles=3,
        max_energy=30,
        max_steps=50
    )
    
    obs, info = env.reset(seed=42)
    
    print("Environment Features:")
    print(f"• Grid Size: {env.grid_size}x{env.grid_size}")
    print(f"• Treasures: {env.num_treasures}")
    print(f"• Obstacles: {env.num_obstacles}")
    print(f"• Max Energy: {env.max_energy}")
    print(f"• Max Steps: {env.max_steps}")
    print(f"• Actions: Up(0), Right(1), Down(2), Left(3), Stay(4)")
    
    print(f"\nObservation Space: {env.observation_space.shape}")
    print("Observation includes:")
    print("• Agent position (x, y)")
    print("• Current energy level")
    print("• Steps remaining")
    print("• Treasure locations (grid)")
    print("• Obstacle locations (grid)")
    
    print(f"\nReward Structure:")
    print("• +100 for collecting a treasure")
    print("• -1 for each step (time penalty)")
    print("• -5 for hitting an obstacle")
    print("• -10 for running out of energy")
    print("• +50 bonus for collecting all treasures")
    
    print(f"\nEnergy Management:")
    print(f"• Moving costs {env.energy_cost_move} energy")
    print(f"• Staying recovers {env.energy_recovery_rate} energy")
    print(f"• Episode ends if energy reaches 0")
    
    # Show initial state
    print(f"\nInitial State:")
    print(f"• Agent Position: {info['agent_pos']}")
    print(f"• Energy: {info['energy']}")
    print(f"• Treasures to Collect: {info['total_treasures']}")
    
    # Visualize initial state
    fig = env.render('matplotlib')
    plt.title("GridWorld Treasure Hunt - Initial State")
    plt.show()
    
    env.close()

if __name__ == "__main__":
    # Demonstrate the environment
    demonstrate_environment()
    
    # Ask user what to run
    print("\nWhat would you like to run?")
    print("1. Q-Learning example")
    print("2. DQN example") 
    print("3. Compare both algorithms")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        run_tabular_example()
    elif choice == "2":
        run_deep_rl_example()
    elif choice == "3":
        compare_algorithms()
    elif choice == "4":
        print("Goodbye!")
    else:
        print("Invalid choice. Running comparison by default...")
        compare_algorithms() 