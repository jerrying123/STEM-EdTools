"""
Training worker thread for running RL algorithms without blocking the GUI
"""
import gymnasium as gym
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from ..algorithms import (QLearning, SARSA, ExpectedSARSA, DoubleQLearning, 
                         MonteCarloControl, NStepSARSA, DQN, DoubleDQN, 
                         DuelingDQN, REINFORCE)
from .. import environments  # Ensure custom environments are registered

class TrainingWorker(QThread):
    # Signals for communicating with the main thread
    progress_updated = pyqtSignal(int)  # Progress percentage
    episode_completed = pyqtSignal(int, dict)  # Episode number, rewards dict
    training_completed = pyqtSignal(dict)  # Final results
    log_message = pyqtSignal(str)  # Log messages
    
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.should_stop = False
        
    def run(self):
        """Main training loop"""
        try:
            import time
            start_time = time.time()
            self.log_message.emit(f"[{time.time():.3f}] Starting training...")
            self.log_message.emit(f"[{time.time():.3f}] Parameters: {self.params}")
            
            # Create environment
            env = gym.make(self.params['environment'])
            self.log_message.emit(f"[{time.time():.3f}] Created environment: {self.params['environment']}")
            
            # Create state bins for discretization
            state_bins = self.create_state_bins(env, self.params['discretization_bins'])
            self.log_message.emit(f"[{time.time():.3f}] Created state bins")
            
            # Initialize algorithms
            algorithms = self.initialize_algorithms()
            self.log_message.emit(f"[{time.time():.3f}] Initialized algorithms: {list(algorithms.keys())}")
            
            # Training parameters
            episodes = self.params['episodes']
            max_steps = self.params['max_steps']
            
            # Track rewards for each algorithm
            rewards_history = {name: [] for name in algorithms.keys()}
            
            # Training loop
            self.log_message.emit(f"[{time.time():.3f}] Starting training loop: {episodes} episodes")
            for episode in range(episodes):
                if self.should_stop:
                    break
                
                episode_start_time = time.time()
                episode_rewards = {}
                
                for algo_name, agent in algorithms.items():
                    if self.should_stop:
                        break
                    
                    # Run episode for this algorithm
                    algo_start_time = time.time()
                    total_reward = self.run_episode(env, agent, state_bins, max_steps, algo_name)
                    algo_elapsed = time.time() - algo_start_time
                    
                    rewards_history[algo_name].append(total_reward)
                    episode_rewards[algo_name] = total_reward
                    
                    if episode < 3:  # Log timing for first few episodes
                        self.log_message.emit(f"[{time.time():.3f}] Episode {episode + 1} {algo_name}: {total_reward:.1f} (took {algo_elapsed:.3f}s)")
                
                # Update progress every episode (Qt queued connections will handle compression)
                progress = int((episode + 1) / episodes * 100)
                self.progress_updated.emit(progress)
                
                # Emit episode completion signal every episode
                signal_start_time = time.time()
                self.episode_completed.emit(episode + 1, rewards_history.copy())
                signal_elapsed = time.time() - signal_start_time
                
                # Periodically yield to Qt event loop for responsive GUI
                if (episode + 1) % 50 == 0:  # Less frequent since we're using queued connections
                    from PyQt6.QtCore import QCoreApplication
                    QCoreApplication.processEvents()
                
                episode_elapsed = time.time() - episode_start_time
                
                # Log progress periodically
                if (episode + 1) % max(1, episodes // 10) == 0:
                    log_msg = f"[{time.time():.3f}] Episode {episode + 1}/{episodes}"
                    for algo_name, reward in episode_rewards.items():
                        log_msg += f" | {algo_name}: {reward:.1f}"
                    log_msg += f" (episode took {episode_elapsed:.3f}s, signal took {signal_elapsed:.3f}s)"
                    self.log_message.emit(log_msg)
                
                # Also log every episode for the first few to debug timing
                if episode < 5:
                    log_msg = f"[{time.time():.3f}] Episode {episode + 1} completed in {episode_elapsed:.3f}s"
                    for algo_name, reward in episode_rewards.items():
                        log_msg += f" | {algo_name}: {reward:.1f}"
                    self.log_message.emit(log_msg)
            
            env.close()
            self.log_message.emit(f"[{time.time():.3f}] Environment closed")
            
            # Prepare final results
            results_start_time = time.time()
            results = {}
            for algo_name, agent in algorithms.items():
                rewards = rewards_history[algo_name]
                results[algo_name] = {
                    'agent': agent,
                    'rewards': rewards,
                    'avg_reward': np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards),
                    'best_reward': max(rewards) if rewards else 0
                }
            results_elapsed = time.time() - results_start_time
            self.log_message.emit(f"[{time.time():.3f}] Results prepared (took {results_elapsed:.3f}s)")
            
            signal_emit_time = time.time()
            self.log_message.emit(f"[{time.time():.3f}] Emitting training completed signal...")
            self.training_completed.emit(results)
            signal_emit_elapsed = time.time() - signal_emit_time
            self.log_message.emit(f"[{time.time():.3f}] Training completed signal emitted (took {signal_emit_elapsed:.3f}s)")
            
            # Ensure thread terminates promptly
            self.quit()
            self.log_message.emit(f"[{time.time():.3f}] Thread quit() called")
            
        except Exception as e:
            self.log_message.emit(f"[{time.time():.3f}] Error during training: {str(e)}")
            import traceback
            self.log_message.emit(f"[{time.time():.3f}] {traceback.format_exc()}")
    
    def stop(self):
        """Stop the training process"""
        self.should_stop = True
    
    def create_state_bins(self, env, num_bins):
        """Create bins for discretizing the state space"""
        if env.spec.id == 'CartPole-v1':
            # CartPole state: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
            state_bounds = [
                (-2.4, 2.4),    # cart position
                (-3.0, 3.0),    # cart velocity  
                (-0.2, 0.2),    # pole angle
                (-3.0, 3.0)     # pole angular velocity
            ]
        elif env.spec.id == 'MountainCar-v0':
            # MountainCar state: [position, velocity]
            state_bounds = [
                (-1.2, 0.6),    # position
                (-0.07, 0.07)   # velocity
            ]
        elif env.spec.id == 'Acrobot-v1':
            # Acrobot has 6-dimensional state space, use reasonable bounds
            state_bounds = [
                (-1.0, 1.0),    # cos(theta1)
                (-1.0, 1.0),    # sin(theta1)
                (-1.0, 1.0),    # cos(theta2)
                (-1.0, 1.0),    # sin(theta2)
                (-4.0, 4.0),    # theta1_dot
                (-9.0, 9.0)     # theta2_dot
            ]
        elif env.spec.id == 'GridWorldTreasure-v0':
            # GridWorldTreasure has a large observation space (position + energy + steps + grids)
            # For tabular methods, we'll only discretize the key features
            grid_size = getattr(env, 'grid_size', 8)
            max_energy = getattr(env, 'max_energy', 100)
            max_steps = getattr(env, 'max_steps', 200)
            
            state_bounds = [
                (0, grid_size - 1),     # agent_x
                (0, grid_size - 1),     # agent_y  
                (0, max_energy),        # energy
                (0, max_steps)          # steps_remaining
            ]
            # Note: For simplicity, we ignore the treasure and obstacle grids in tabular methods
            # Deep RL methods will use the full observation space
        else:
            # For other environments, use the original logic but clip infinite values
            low = env.observation_space.low
            high = env.observation_space.high
            
            # Replace infinite values with reasonable bounds
            low = np.where(np.isfinite(low), low, -10.0)
            high = np.where(np.isfinite(high), high, 10.0)
            
            state_bounds = list(zip(low, high))
        
        bins = [np.linspace(low, high, num_bins) for low, high in state_bounds]
        return bins
    
    def discretize_state(self, observation, bins):
        """Discretize continuous state space into bins"""
        # For GridWorldTreasure, only discretize the first 4 elements (position, energy, steps)
        if len(observation) > len(bins):
            # This is likely GridWorldTreasure with a large observation space
            # Only discretize the key features we defined in create_state_bins
            key_features = observation[:len(bins)]
            return tuple(np.digitize(key_features[i], bins[i]) for i in range(len(bins)))
        else:
            # Standard discretization for other environments
            return tuple(np.digitize(observation[i], bins[i]) for i in range(len(observation)))
    
    def initialize_algorithms(self):
        """Initialize the selected algorithms"""
        algorithms = {}
        
        # Get action space size (assuming discrete action space)
        env = gym.make(self.params['environment'])
        action_space_size = env.action_space.n
        env.close()
        
        algorithm_choice = self.params['algorithm']
        
        # Common parameters
        common_params = {
            'action_space_size': action_space_size,
            'learning_rate': self.params['learning_rate'],
            'discount_factor': self.params['discount_factor'],
            'epsilon': self.params['epsilon']
        }
        
        if algorithm_choice == "Q-Learning" or algorithm_choice == "Compare All":
            algorithms["Q-Learning"] = QLearning(**common_params)
        
        if algorithm_choice == "SARSA" or algorithm_choice == "Compare All":
            algorithms["SARSA"] = SARSA(**common_params)
        
        if algorithm_choice == "Expected SARSA" or algorithm_choice == "Compare All":
            algorithms["Expected SARSA"] = ExpectedSARSA(**common_params)
        
        if algorithm_choice == "Double Q-Learning" or algorithm_choice == "Compare All":
            algorithms["Double Q-Learning"] = DoubleQLearning(**common_params)
        
        if algorithm_choice == "Monte Carlo Control" or algorithm_choice == "Compare All":
            algorithms["Monte Carlo Control"] = MonteCarloControl(**common_params)
        
        if algorithm_choice == "n-step SARSA" or algorithm_choice in ["Compare Tabular", "Compare All"]:
            # Add n_steps parameter for n-step SARSA
            n_step_params = common_params.copy()
            n_step_params['n_steps'] = self.params.get('n_steps', 3)
            algorithms["n-step SARSA"] = NStepSARSA(**n_step_params)
        
        # Deep RL algorithms - use continuous state space
        if algorithm_choice in ["DQN", "Double DQN", "Dueling DQN", "REINFORCE", "Compare Deep RL", "Compare All"]:
            # Get state size from environment
            env_temp = gym.make(self.params['environment'])
            state_size = env_temp.observation_space.shape[0]
            env_temp.close()
            
            # Common deep RL parameters
            deep_params = {
                'state_size': state_size,
                'action_size': action_space_size,
                'learning_rate': self.params['learning_rate'],
                'discount_factor': self.params['discount_factor'],
                'epsilon': self.params['epsilon'],
                'epsilon_decay': self.params.get('epsilon_decay', 0.995),
                'hidden_size': self.params.get('hidden_size', 64)
            }
            
            if algorithm_choice == "DQN" or algorithm_choice in ["Compare Deep RL", "Compare All"]:
                algorithms["DQN"] = DQN(**deep_params)
            
            if algorithm_choice == "Double DQN" or algorithm_choice in ["Compare Deep RL", "Compare All"]:
                double_dqn_params = deep_params.copy()
                double_dqn_params['target_update_frequency'] = self.params.get('target_update_frequency', 100)
                algorithms["Double DQN"] = DoubleDQN(**double_dqn_params)
            
            if algorithm_choice == "Dueling DQN" or algorithm_choice in ["Compare Deep RL", "Compare All"]:
                dueling_dqn_params = deep_params.copy()
                dueling_dqn_params['target_update_frequency'] = self.params.get('target_update_frequency', 100)
                algorithms["Dueling DQN"] = DuelingDQN(**dueling_dqn_params)
            
            if algorithm_choice == "REINFORCE" or algorithm_choice in ["Compare Deep RL", "Compare All"]:
                # REINFORCE doesn't need epsilon parameters
                reinforce_params = {
                    'state_size': state_size,
                    'action_size': action_space_size,
                    'learning_rate': self.params['learning_rate'],
                    'discount_factor': self.params['discount_factor'],
                    'hidden_size': self.params.get('hidden_size', 64)
                }
                algorithms["REINFORCE"] = REINFORCE(**reinforce_params)
        
        return algorithms
    
    def run_episode(self, env, agent, state_bins, max_steps, algo_name):
        """Run a single episode for the given agent"""
        state, _ = env.reset()
        
        # Deep RL algorithms use continuous states, tabular methods use discretized states
        is_deep_rl = algo_name in ["DQN", "Double DQN", "Dueling DQN", "REINFORCE"]
        if not is_deep_rl:
            state = self.discretize_state(state, state_bins)
        
        total_reward = 0
        
        # Initialize episode for algorithms that need it
        if hasattr(agent, 'start_episode'):
            agent.start_episode()
        
        if algo_name in ["SARSA", "n-step SARSA"]:
            # SARSA-based algorithms need to select the first action
            action = agent.select_action(state)
        
        for step in range(max_steps):
            if self.should_stop:
                break
            
            if algo_name not in ["SARSA", "n-step SARSA"]:
                action = agent.select_action(state)
            
            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Discretize state for tabular methods
            if not is_deep_rl:
                next_state = self.discretize_state(next_state, state_bins)
            
            done = terminated or truncated
            
            # Update agent based on algorithm type
            if algo_name in ["Q-Learning", "Expected SARSA", "Double Q-Learning", "Monte Carlo Control"]:
                agent.update(state, action, reward, next_state)
            elif algo_name in ["SARSA", "n-step SARSA"]:
                next_action = agent.select_action(next_state)
                agent.update(state, action, reward, next_state, next_action)
                action = next_action
            elif is_deep_rl:
                # Deep RL algorithms need the 'done' flag
                agent.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # End episode for algorithms that need it
        if hasattr(agent, 'end_episode'):
            if algo_name == "n-step SARSA":
                agent.end_episode(final_state=state)
            else:
                agent.end_episode()
        
        return total_reward 