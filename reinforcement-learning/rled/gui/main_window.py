"""
Main GUI window for RLEd - Reinforcement Learning Education Tool
"""
import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QGridLayout, QLabel, QSpinBox, 
                            QDoubleSpinBox, QComboBox, QPushButton, QTextEdit,
                            QProgressBar, QGroupBox, QTabWidget, QSplitter, QCheckBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

from ..algorithms import QLearning, SARSA
from .training_worker import TrainingWorker

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RLEd - Reinforcement Learning Education Tool")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize variables
        self.training_worker = None
        self.training_results = {}
        self.current_rewards_data = {}
        
        # Setup GUI update timer for smooth real-time updates
        from PyQt6.QtCore import QTimer
        self.gui_update_timer = QTimer()
        self.gui_update_timer.timeout.connect(self.process_pending_updates)
        self.gui_update_timer.setInterval(50)  # Update GUI every 50ms
        self.pending_plot_update = False
        
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Controls
        left_panel = self.create_control_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - Visualization and results
        right_panel = self.create_visualization_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([400, 1000])
        
    def create_control_panel(self):
        """Create the control panel with parameter settings"""
        control_widget = QWidget()
        layout = QVBoxLayout(control_widget)
        
        # Title
        title = QLabel("Training Parameters")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        layout.addWidget(title)
        
        # Algorithm selection
        algo_group = QGroupBox("Algorithm")
        algo_layout = QVBoxLayout(algo_group)
        
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems([
            # Tabular methods
            "Q-Learning", 
            "SARSA", 
            "Expected SARSA",
            "Double Q-Learning",
            "Monte Carlo Control",
            "n-step SARSA",
            # Deep RL methods
            "DQN",
            "Double DQN",
            "Dueling DQN", 
            "REINFORCE",
            # Comparison options
            "Compare Tabular",
            "Compare Deep RL",
            "Compare All"
        ])
        algo_layout.addWidget(self.algorithm_combo)
        layout.addWidget(algo_group)
        
        # Environment settings
        env_group = QGroupBox("Environment")
        env_layout = QGridLayout(env_group)
        
        env_layout.addWidget(QLabel("Environment:"), 0, 0)
        self.env_combo = QComboBox()
        self.env_combo.addItems([
            "CartPole-v1", 
            "MountainCar-v0", 
            "Acrobot-v1",
            "GridWorldTreasure-v0"
        ])
        env_layout.addWidget(self.env_combo, 0, 1)
        
        env_layout.addWidget(QLabel("Discretization Bins:"), 1, 0)
        self.bins_spinbox = QSpinBox()
        self.bins_spinbox.setRange(5, 50)
        self.bins_spinbox.setValue(10)
        env_layout.addWidget(self.bins_spinbox, 1, 1)
        
        layout.addWidget(env_group)
        
        # Training parameters
        training_group = QGroupBox("Training Parameters")
        training_layout = QGridLayout(training_group)
        
        training_layout.addWidget(QLabel("Episodes:"), 0, 0)
        self.episodes_spinbox = QSpinBox()
        self.episodes_spinbox.setRange(10, 10000)
        self.episodes_spinbox.setValue(500)
        training_layout.addWidget(self.episodes_spinbox, 0, 1)
        
        training_layout.addWidget(QLabel("Max Steps:"), 1, 0)
        self.max_steps_spinbox = QSpinBox()
        self.max_steps_spinbox.setRange(50, 2000)
        self.max_steps_spinbox.setValue(500)
        training_layout.addWidget(self.max_steps_spinbox, 1, 1)
        
        layout.addWidget(training_group)
        
        # Hyperparameters
        hyper_group = QGroupBox("Hyperparameters")
        hyper_layout = QGridLayout(hyper_group)
        
        hyper_layout.addWidget(QLabel("Learning Rate (α):"), 0, 0)
        self.learning_rate_spinbox = QDoubleSpinBox()
        self.learning_rate_spinbox.setRange(0.001, 1.0)
        self.learning_rate_spinbox.setSingleStep(0.01)
        self.learning_rate_spinbox.setValue(0.1)
        self.learning_rate_spinbox.setDecimals(3)
        hyper_layout.addWidget(self.learning_rate_spinbox, 0, 1)
        
        hyper_layout.addWidget(QLabel("Discount Factor (γ):"), 1, 0)
        self.discount_spinbox = QDoubleSpinBox()
        self.discount_spinbox.setRange(0.1, 1.0)
        self.discount_spinbox.setSingleStep(0.01)
        self.discount_spinbox.setValue(0.99)
        self.discount_spinbox.setDecimals(3)
        hyper_layout.addWidget(self.discount_spinbox, 1, 1)
        
        hyper_layout.addWidget(QLabel("Epsilon (ε):"), 2, 0)
        self.epsilon_spinbox = QDoubleSpinBox()
        self.epsilon_spinbox.setRange(0.0, 1.0)
        self.epsilon_spinbox.setSingleStep(0.01)
        self.epsilon_spinbox.setValue(0.1)
        self.epsilon_spinbox.setDecimals(3)
        hyper_layout.addWidget(self.epsilon_spinbox, 2, 1)
        
        hyper_layout.addWidget(QLabel("n-steps (for n-step SARSA):"), 3, 0)
        self.n_steps_spinbox = QSpinBox()
        self.n_steps_spinbox.setRange(1, 10)
        self.n_steps_spinbox.setValue(3)
        hyper_layout.addWidget(self.n_steps_spinbox, 3, 1)
        
        layout.addWidget(hyper_group)
        
        # Deep RL parameters
        deep_group = QGroupBox("Deep RL Parameters")
        deep_layout = QGridLayout(deep_group)
        
        deep_layout.addWidget(QLabel("Hidden Layer Size:"), 0, 0)
        self.hidden_size_spinbox = QSpinBox()
        self.hidden_size_spinbox.setRange(16, 512)
        self.hidden_size_spinbox.setValue(64)
        deep_layout.addWidget(self.hidden_size_spinbox, 0, 1)
        
        deep_layout.addWidget(QLabel("Batch Size:"), 1, 0)
        self.batch_size_spinbox = QSpinBox()
        self.batch_size_spinbox.setRange(8, 128)
        self.batch_size_spinbox.setValue(32)
        deep_layout.addWidget(self.batch_size_spinbox, 1, 1)
        
        deep_layout.addWidget(QLabel("Target Update Freq:"), 2, 0)
        self.target_update_spinbox = QSpinBox()
        self.target_update_spinbox.setRange(10, 1000)
        self.target_update_spinbox.setValue(100)
        deep_layout.addWidget(self.target_update_spinbox, 2, 1)
        
        deep_layout.addWidget(QLabel("Epsilon Decay:"), 3, 0)
        self.epsilon_decay_spinbox = QDoubleSpinBox()
        self.epsilon_decay_spinbox.setRange(0.9, 0.999)
        self.epsilon_decay_spinbox.setSingleStep(0.001)
        self.epsilon_decay_spinbox.setValue(0.995)
        self.epsilon_decay_spinbox.setDecimals(3)
        deep_layout.addWidget(self.epsilon_decay_spinbox, 3, 1)
        
        layout.addWidget(deep_group)
        
        # Control buttons
        button_layout = QVBoxLayout()
        
        self.start_button = QPushButton("Start Training")
        self.start_button.clicked.connect(self.start_training)
        button_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop Training")
        self.stop_button.clicked.connect(self.stop_training)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)
        
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_training)
        button_layout.addWidget(self.reset_button)
        
        layout.addLayout(button_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # Status
        self.status_label = QLabel("Ready to train")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        
        return control_widget
    
    def create_visualization_panel(self):
        """Create the visualization panel with plots and results"""
        viz_widget = QWidget()
        layout = QVBoxLayout(viz_widget)
        
        # Create tab widget for different views
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Training progress tab
        self.create_training_tab()
        
        # Q-table visualization tab
        self.create_qtable_tab()
        
        # Policy animation tab
        self.create_policy_animation_tab()
        
        # Results tab
        self.create_results_tab()
        
        return viz_widget
    
    def create_training_tab(self):
        """Create the training progress visualization tab"""
        training_widget = QWidget()
        layout = QVBoxLayout(training_widget)
        
        # Plot controls
        controls_layout = QHBoxLayout()
        
        # Checkbox for showing raw data
        self.show_raw_data_cb = QCheckBox("Show Raw Data")
        self.show_raw_data_cb.setChecked(False)  # Default to off
        self.show_raw_data_cb.stateChanged.connect(self.refresh_training_plot)
        controls_layout.addWidget(self.show_raw_data_cb)
        
        # Checkbox for showing moving average
        self.show_moving_avg_cb = QCheckBox("Show Moving Average")
        self.show_moving_avg_cb.setChecked(True)  # Default to on
        self.show_moving_avg_cb.stateChanged.connect(self.refresh_training_plot)
        controls_layout.addWidget(self.show_moving_avg_cb)
        
        # Moving average window size
        controls_layout.addWidget(QLabel("Window Size:"))
        self.window_size_spinbox = QSpinBox()
        self.window_size_spinbox.setRange(5, 200)
        self.window_size_spinbox.setValue(50)
        self.window_size_spinbox.valueChanged.connect(self.refresh_training_plot)
        controls_layout.addWidget(self.window_size_spinbox)
        
        # Add stretch to push controls to the left
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        
        # Matplotlib figure for training progress
        self.training_figure = Figure(figsize=(10, 6))
        self.training_canvas = FigureCanvas(self.training_figure)
        layout.addWidget(self.training_canvas)
        
        # Store current rewards data for refreshing
        self.current_rewards_data = {}
        
        self.tab_widget.addTab(training_widget, "Training Progress")
    
    def create_qtable_tab(self):
        """Create the Q-table visualization tab"""
        qtable_widget = QWidget()
        layout = QVBoxLayout(qtable_widget)
        
        # Interactive Q-table visualization
        from .interactive_qtable import InteractiveQTableWidget
        self.interactive_qtable = InteractiveQTableWidget()
        layout.addWidget(self.interactive_qtable)
        
        self.tab_widget.addTab(qtable_widget, "Q-Table Visualization")
    
    def create_policy_animation_tab(self):
        """Create the policy animation tab"""
        animation_widget = QWidget()
        layout = QVBoxLayout(animation_widget)
        
        # Policy animation widget
        from .policy_animation import PolicyAnimationWidget
        self.policy_animation = PolicyAnimationWidget()
        layout.addWidget(self.policy_animation)
        
        self.tab_widget.addTab(animation_widget, "Policy Animation")
    
    def create_results_tab(self):
        """Create the results and logs tab"""
        results_widget = QWidget()
        layout = QVBoxLayout(results_widget)
        
        # Text area for detailed results
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFont(QFont("Courier", 10))
        layout.addWidget(self.results_text)
        
        self.tab_widget.addTab(results_widget, "Results & Logs")
    
    def start_training(self):
        """Start the training process"""
        if self.training_worker is not None and self.training_worker.isRunning():
            return
        
        # Get parameters from UI
        params = self.get_training_parameters()
        
        # Create and start training worker
        self.training_worker = TrainingWorker(params)
        # Use queued connections for better performance - Qt will automatically compress redundant signals
        from PyQt6.QtCore import Qt
        self.training_worker.progress_updated.connect(self.update_progress, Qt.ConnectionType.QueuedConnection)
        self.training_worker.episode_completed.connect(self.update_training_plot, Qt.ConnectionType.QueuedConnection)
        self.training_worker.training_completed.connect(self.training_finished, Qt.ConnectionType.QueuedConnection)
        self.training_worker.log_message.connect(self.add_log_message, Qt.ConnectionType.QueuedConnection)
        
        self.training_worker.start()
        
        # Update UI state
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("Training in progress...")
        self.progress_bar.setValue(0)
        
        # Clear previous results
        self.results_text.clear()
        self.training_results = {}
    
    def stop_training(self):
        """Stop the training process"""
        if self.training_worker is not None:
            self.training_worker.stop()
            self.training_worker.wait()
        
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Training stopped")
    
    def reset_training(self):
        """Reset the training and clear results"""
        self.stop_training()
        
        # Clear plots and stored data
        self.training_figure.clear()
        self.training_canvas.draw()
        self.current_rewards_data = {}
        
        # Reset interactive Q-table
        self.interactive_qtable.q_tables = {}
        self.interactive_qtable.update_visualization()
        
        # Reset policy animation
        if hasattr(self, 'policy_animation'):
            self.policy_animation.animation_data = []
            self.policy_animation.status_label.setText("No animation recorded")
            self.policy_animation.play_button.setEnabled(False)
            self.policy_animation.stop_button.setEnabled(False)
            self.policy_animation.frame_slider.setEnabled(False)
            self.policy_animation.save_button.setEnabled(False)
        
        # Clear results
        self.results_text.clear()
        self.training_results = {}
        
        self.progress_bar.setValue(0)
        self.status_label.setText("Ready to train")
    
    def get_training_parameters(self):
        """Get training parameters from UI controls"""
        return {
            'algorithm': self.algorithm_combo.currentText(),
            'environment': self.env_combo.currentText(),
            'episodes': self.episodes_spinbox.value(),
            'max_steps': self.max_steps_spinbox.value(),
            'learning_rate': self.learning_rate_spinbox.value(),
            'discount_factor': self.discount_spinbox.value(),
            'epsilon': self.epsilon_spinbox.value(),
            'discretization_bins': self.bins_spinbox.value(),
            'n_steps': self.n_steps_spinbox.value(),
            # Deep RL parameters
            'hidden_size': self.hidden_size_spinbox.value(),
            'batch_size': self.batch_size_spinbox.value(),
            'target_update_frequency': self.target_update_spinbox.value(),
            'epsilon_decay': self.epsilon_decay_spinbox.value()
        }
    
    def update_progress(self, progress):
        """Update the progress bar"""
        self.progress_bar.setValue(progress)
        # Force immediate GUI update
        from PyQt6.QtCore import QCoreApplication
        QCoreApplication.processEvents()
        if progress in [1, 10, 25, 50, 75, 90, 100]:  # Log at key milestones
            import time
            self.add_log_message(f"Progress updated to {progress}%")
    
    def update_training_plot(self, episode, rewards):
        """Update the training progress plot"""
        # Store current data for refreshing
        self.current_rewards_data = rewards
        
        # Mark that we have a pending plot update (timer will handle actual plotting)
        self.pending_plot_update = True
        
        # Start the GUI update timer if not already running
        if not self.gui_update_timer.isActive():
            self.gui_update_timer.start()
    
    def process_pending_updates(self):
        """Process pending GUI updates (called by timer)"""
        if self.pending_plot_update and self.current_rewards_data:
            self.plot_training_data()
            self.pending_plot_update = False
        
        # Stop timer if no more updates pending
        if not self.pending_plot_update:
            self.gui_update_timer.stop()
    
    def refresh_training_plot(self):
        """Refresh the training plot with current settings"""
        if self.current_rewards_data:
            self.plot_training_data()
    
    def plot_training_data(self):
        """Plot the training data based on current settings"""
        try:
            self.training_figure.clear()
            ax = self.training_figure.add_subplot(111)
            
            show_raw = self.show_raw_data_cb.isChecked()
            show_avg = self.show_moving_avg_cb.isChecked()
            window_size = self.window_size_spinbox.value()
            
            for algo_name, reward_history in self.current_rewards_data.items():
                if not reward_history:  # Skip empty reward histories
                    continue
                    
                # Ensure arrays have the same length
                episodes = list(range(1, len(reward_history) + 1))
                
                # Ensure both arrays have the same length (defensive programming)
                min_length = min(len(episodes), len(reward_history))
                episodes = episodes[:min_length]
                reward_history_safe = reward_history[:min_length]
                
                # Plot raw data if enabled
                if show_raw:
                    ax.plot(episodes, reward_history_safe, label=f'{algo_name} (Raw)', 
                           alpha=0.4, linewidth=1)
                
                # Add moving average if enabled
                if show_avg and len(reward_history_safe) > 10:
                    # Use user-specified window size, but cap it at reasonable values
                    actual_window_size = min(window_size, len(reward_history_safe) // 2, 200)
                    actual_window_size = max(actual_window_size, 5)  # Minimum window size
                    
                    if actual_window_size > 0:
                        moving_avg = np.convolve(reward_history_safe, 
                                               np.ones(actual_window_size)/actual_window_size, 
                                               mode='valid')
                        # Ensure the episodes array matches the moving average length exactly
                        avg_episodes = list(range(actual_window_size, actual_window_size + len(moving_avg)))
                        
                        # Double-check array lengths match
                        if len(avg_episodes) == len(moving_avg):
                            label = f'{algo_name}' if not show_raw else f'{algo_name} (Avg)'
                            ax.plot(avg_episodes, moving_avg, label=label, linewidth=2)
            
            ax.set_xlabel('Episode')
            ax.set_ylabel('Total Reward')
            ax.set_title('Training Progress')
            
            # Only show legend if there are lines to show
            if ax.get_lines():
                ax.legend()
            
            ax.grid(True, alpha=0.3)
            
            self.training_canvas.draw()
            
        except Exception as e:
            # If plotting fails, just skip this update to prevent crashes
            print(f"Warning: Failed to update training plot: {e}")
            pass
    
    def training_finished(self, results):
        """Handle training completion"""
        import time
        start_time = time.time()
        
        self.training_results = results
        
        # Update UI state immediately
        ui_start_time = time.time()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Training completed")
        self.progress_bar.setValue(100)
        ui_elapsed = time.time() - ui_start_time
        
        # Log completion for debugging
        self.add_log_message(f"[{time.time():.3f}] Training worker finished - updating visualizations... (UI update took {ui_elapsed:.3f}s)")
        
        # Update Q-table visualization
        try:
            qtable_start_time = time.time()
            self.update_qtable_visualization()
            qtable_elapsed = time.time() - qtable_start_time
            self.add_log_message(f"[{time.time():.3f}] Q-table visualization updated (took {qtable_elapsed:.3f}s)")
        except Exception as e:
            self.add_log_message(f"[{time.time():.3f}] Error updating Q-table: {e}")
        
        # Update policy animation widget
        try:
            policy_start_time = time.time()
            self.update_policy_animation()
            policy_elapsed = time.time() - policy_start_time
            self.add_log_message(f"[{time.time():.3f}] Policy animation updated (took {policy_elapsed:.3f}s)")
        except Exception as e:
            self.add_log_message(f"[{time.time():.3f}] Error updating policy animation: {e}")
        
        # Add final results to log
        try:
            results_start_time = time.time()
            self.add_final_results()
            results_elapsed = time.time() - results_start_time
            total_elapsed = time.time() - start_time
            self.add_log_message(f"[{time.time():.3f}] All post-training updates completed (results took {results_elapsed:.3f}s, total took {total_elapsed:.3f}s)")
        except Exception as e:
            self.add_log_message(f"[{time.time():.3f}] Error adding final results: {e}")
    
    def update_qtable_visualization(self):
        """Update the Q-table visualization"""
        if not self.training_results:
            return
        
        # Extract Q-tables from training results
        q_tables = {}
        for algo_name, results in self.training_results.items():
            agent = results['agent']
            if hasattr(agent, 'q_table') and agent.q_table:
                q_tables[algo_name] = agent.q_table
        
        if q_tables:
            # Get environment info
            params = self.get_training_parameters()
            env_name = params['environment']
            discretization_bins = params['discretization_bins']
            
            # Get state bounds from environment (we'll need to create a temporary env)
            import gymnasium as gym
            temp_env = gym.make(env_name)
            state_bounds = []
            if hasattr(temp_env.observation_space, 'low') and hasattr(temp_env.observation_space, 'high'):
                for i in range(len(temp_env.observation_space.low)):
                    low = temp_env.observation_space.low[i]
                    high = temp_env.observation_space.high[i]
                    # Handle infinite bounds
                    if np.isinf(low):
                        low = -10.0
                    if np.isinf(high):
                        high = 10.0
                    state_bounds.append((low, high))
            temp_env.close()
            
            # Update the interactive Q-table widget
            self.interactive_qtable.update_q_tables(
                q_tables, env_name, state_bounds, discretization_bins
            )
    
    def update_policy_animation(self):
        """Update the policy animation widget"""
        if not self.training_results:
            return
        
        # Get environment info
        params = self.get_training_parameters()
        env_name = params['environment']
        
        # Update the policy animation widget
        self.policy_animation.update_policy_data(self.training_results, env_name)
    
    def format_state_label(self, state, env_name):
        """Format state tuple into meaningful label based on environment"""
        if env_name == "CartPole-v1":
            # state = (cart_pos_bin, cart_vel_bin, pole_angle_bin, pole_vel_bin)
            return f"({state[0]},{state[1]},{state[2]},{state[3]})"
        elif env_name == "MountainCar-v0":
            # state = (position_bin, velocity_bin)
            return f"({state[0]},{state[1]})"
        elif env_name == "Acrobot-v1":
            # state = (cos_θ1, sin_θ1, cos_θ2, sin_θ2, θ1_dot, θ2_dot)
            return f"({state[0]},{state[1]},{state[2]},{state[3]},{state[4]},{state[5]})"
        else:
            return str(state)
    
    def get_action_labels(self, env_name, n_actions):
        """Get meaningful action labels based on environment"""
        if env_name == "CartPole-v1":
            return ["Left", "Right"]
        elif env_name == "MountainCar-v0":
            return ["Left", "None", "Right"]
        elif env_name == "Acrobot-v1":
            return ["CCW", "None", "CW"]
        else:
            return [f"Action {i}" for i in range(n_actions)]
    
    def get_state_dimension_label(self, env_name):
        """Get meaningful state dimension label based on environment"""
        if env_name == "CartPole-v1":
            return "State (Cart Pos, Cart Vel, Pole Angle, Pole Vel)"
        elif env_name == "MountainCar-v0":
            return "State (Position, Velocity)"
        elif env_name == "Acrobot-v1":
            return "State (cos θ₁, sin θ₁, cos θ₂, sin θ₂, θ₁̇, θ₂̇)"
        else:
            return "State Dimensions"
    
    def add_log_message(self, message):
        """Add a message to the results log"""
        import time
        # If message doesn't already have a timestamp, add one
        if not message.startswith('[') or ']' not in message[:15]:
            message = f"[{time.time():.3f}] {message}"
        self.results_text.append(message)
    
    def add_final_results(self):
        """Add final training results to the log"""
        if not self.training_results:
            return
        
        self.results_text.append("\n" + "="*50)
        self.results_text.append("FINAL TRAINING RESULTS")
        self.results_text.append("="*50)
        
        for algo_name, results in self.training_results.items():
            self.results_text.append(f"\n{algo_name}:")
            self.results_text.append(f"  Average Reward (last 100 episodes): {results['avg_reward']:.2f}")
            self.results_text.append(f"  Best Episode Reward: {results['best_reward']:.2f}")
            
            # Handle different algorithm types
            agent = results['agent']
            if hasattr(agent, 'q_table') and agent.q_table:
                # Tabular methods
                self.results_text.append(f"  Q-table Size: {len(agent.q_table)} states")
            elif hasattr(agent, 'q_network'):
                # Deep RL methods
                param_count = sum(p.numel() for p in agent.q_network.parameters())
                self.results_text.append(f"  Network Parameters: {param_count:,}")
                if hasattr(agent, 'memory'):
                    self.results_text.append(f"  Experience Buffer Size: {len(agent.memory)}")
                if hasattr(agent, 'epsilon'):
                    self.results_text.append(f"  Final Epsilon: {agent.epsilon:.4f}")
            elif hasattr(agent, 'policy_network'):
                # Policy gradient methods (REINFORCE)
                param_count = sum(p.numel() for p in agent.policy_network.parameters())
                self.results_text.append(f"  Policy Network Parameters: {param_count:,}")
            
            self.results_text.append(f"  Total Episodes: {len(results['rewards'])}")

def main():
    """Main function to run the GUI application"""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 