"""
Policy Animation Widget - Visualizes the physical system with policy execution
"""
import numpy as np
import gymnasium as gym
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, 
                            QLabel, QSlider, QCheckBox, QComboBox, QPushButton,
                            QGroupBox, QSpinBox, QFileDialog, QMessageBox)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Circle
from matplotlib.animation import FuncAnimation
import pickle
import json
import os

class PolicyAnimationWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.env = None
        self.agent = None
        self.env_name = ""
        self.animation_data = []
        self.current_frame = 0
        self.is_playing = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.training_results = {}
        
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the policy animation UI"""
        layout = QVBoxLayout(self)
        
        # Control panel
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel)
        
        # Animation area
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Animation controls
        animation_controls = self.create_animation_controls()
        layout.addWidget(animation_controls)
        
    def create_control_panel(self):
        """Create the control panel for policy selection and recording"""
        control_widget = QWidget()
        layout = QVBoxLayout(control_widget)
        
        # Title
        title = QLabel("Policy Animation & Visualization")
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(title)
        
        # Main controls layout
        main_controls = QHBoxLayout()
        
        # Policy selection
        policy_group = self.create_policy_selection()
        main_controls.addWidget(policy_group)
        
        # Recording controls
        recording_group = self.create_recording_controls()
        main_controls.addWidget(recording_group)
        
        # Visualization options
        viz_group = self.create_visualization_options()
        main_controls.addWidget(viz_group)
        
        layout.addLayout(main_controls)
        
        return control_widget
    
    def create_policy_selection(self):
        """Create policy selection controls"""
        group = QGroupBox("Policy Selection")
        layout = QVBoxLayout(group)
        
        # Algorithm selection
        layout.addWidget(QLabel("Algorithm:"))
        self.algorithm_combo = QComboBox()
        layout.addWidget(self.algorithm_combo)
        
        # Episode selection for recording
        layout.addWidget(QLabel("Episodes to Record:"))
        self.episodes_spin = QSpinBox()
        self.episodes_spin.setRange(1, 10)
        self.episodes_spin.setValue(3)
        layout.addWidget(self.episodes_spin)
        
        # Max steps per episode
        layout.addWidget(QLabel("Max Steps per Episode:"))
        self.max_steps_spin = QSpinBox()
        self.max_steps_spin.setRange(50, 1000)
        self.max_steps_spin.setValue(200)
        layout.addWidget(self.max_steps_spin)
        
        return group
    
    def create_recording_controls(self):
        """Create recording and playback controls"""
        group = QGroupBox("Recording & Playback")
        layout = QVBoxLayout(group)
        
        # Record button
        self.record_button = QPushButton("Record Policy")
        self.record_button.clicked.connect(self.record_policy)
        layout.addWidget(self.record_button)
        
        # Save/Load buttons
        button_layout = QHBoxLayout()
        
        self.save_button = QPushButton("Save Animation")
        self.save_button.clicked.connect(self.save_animation)
        self.save_button.setEnabled(False)
        button_layout.addWidget(self.save_button)
        
        self.load_button = QPushButton("Load Animation")
        self.load_button.clicked.connect(self.load_animation)
        button_layout.addWidget(self.load_button)
        
        layout.addLayout(button_layout)
        
        # Status
        self.status_label = QLabel("No animation recorded")
        layout.addWidget(self.status_label)
        
        return group
    
    def create_visualization_options(self):
        """Create visualization option controls"""
        group = QGroupBox("Visualization Options")
        layout = QVBoxLayout(group)
        
        # Show trajectory
        self.show_trajectory_cb = QCheckBox("Show Trajectory")
        self.show_trajectory_cb.setChecked(True)
        layout.addWidget(self.show_trajectory_cb)
        
        # Show Q-values
        self.show_qvalues_cb = QCheckBox("Show Q-Values")
        self.show_qvalues_cb.setChecked(False)
        layout.addWidget(self.show_qvalues_cb)
        
        # Show state info
        self.show_state_info_cb = QCheckBox("Show State Info")
        self.show_state_info_cb.setChecked(True)
        layout.addWidget(self.show_state_info_cb)
        
        # Animation speed
        layout.addWidget(QLabel("Animation Speed:"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(1, 10)
        self.speed_slider.setValue(5)
        self.speed_slider.valueChanged.connect(self.update_animation_speed)
        layout.addWidget(self.speed_slider)
        
        self.speed_label = QLabel("5x")
        layout.addWidget(self.speed_label)
        
        return group
    
    def create_animation_controls(self):
        """Create animation playback controls"""
        control_widget = QWidget()
        layout = QHBoxLayout(control_widget)
        
        # Play/Pause button
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_playback)
        self.play_button.setEnabled(False)
        layout.addWidget(self.play_button)
        
        # Stop button
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_playback)
        self.stop_button.setEnabled(False)
        layout.addWidget(self.stop_button)
        
        # Frame slider
        layout.addWidget(QLabel("Frame:"))
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.valueChanged.connect(self.seek_frame)
        self.frame_slider.setEnabled(False)
        layout.addWidget(self.frame_slider)
        
        # Frame info
        self.frame_label = QLabel("0 / 0")
        layout.addWidget(self.frame_label)
        
        return control_widget
    
    def update_policy_data(self, training_results, env_name):
        """Update the available policies from training results"""
        self.training_results = training_results
        self.env_name = env_name
        
        # Update algorithm combo
        self.algorithm_combo.clear()
        if training_results:
            self.algorithm_combo.addItems(list(training_results.keys()))
            self.record_button.setEnabled(True)
        else:
            self.record_button.setEnabled(False)
    
    def record_policy(self):
        """Record the policy execution for animation"""
        if not self.training_results:
            QMessageBox.warning(self, "Warning", "No trained policies available!")
            return
        
        algo_name = self.algorithm_combo.currentText()
        if algo_name not in self.training_results:
            return
        
        agent = self.training_results[algo_name]['agent']
        
        # Create environment
        self.env = gym.make(self.env_name, render_mode=None)
        
        # Create state bins for discretization (same as in training)
        self.state_bins = self.create_state_bins(self.env, 10)  # Use same discretization as training
        
        # Record episodes
        self.animation_data = []
        episodes_to_record = self.episodes_spin.value()
        max_steps = self.max_steps_spin.value()
        
        self.status_label.setText(f"Recording {episodes_to_record} episodes...")
        
        for episode in range(episodes_to_record):
            episode_data = self.record_episode(agent, max_steps, episode)
            self.animation_data.extend(episode_data)
        
        self.env.close()
        
        # Enable playback controls
        if self.animation_data:
            self.play_button.setEnabled(True)
            self.stop_button.setEnabled(True)
            self.frame_slider.setEnabled(True)
            self.save_button.setEnabled(True)
            
            self.frame_slider.setRange(0, len(self.animation_data) - 1)
            self.frame_label.setText(f"0 / {len(self.animation_data)}")
            
            self.status_label.setText(f"Recorded {len(self.animation_data)} frames from {episodes_to_record} episodes")
            
            # Show first frame
            self.current_frame = 0
            self.visualize_frame(0)
    
    def record_episode(self, agent, max_steps, episode_num):
        """Record a single episode"""
        episode_data = []
        state, _ = self.env.reset()
        
        # Determine if this is a Deep RL agent
        is_deep_rl = hasattr(agent, 'q_network') or hasattr(agent, 'policy_network')
        
        for step in range(max_steps):
            # For tabular methods, discretize state; for Deep RL, use continuous state
            if is_deep_rl:
                agent_state = state
            else:
                agent_state = self.discretize_state(state, self.state_bins)
            
            # Get action from agent (temporarily set epsilon to 0 for greedy policy)
            original_epsilon = getattr(agent, 'epsilon', None)
            if original_epsilon is not None:
                agent.epsilon = 0.0
            action = agent.select_action(agent_state)
            if original_epsilon is not None:
                agent.epsilon = original_epsilon  # Restore original epsilon
            
            # Get Q-values if available
            q_values = None
            if hasattr(agent, 'q_table') and not is_deep_rl:
                # Tabular methods
                if agent_state in agent.q_table:
                    q_values = agent.q_table[agent_state]
            elif hasattr(agent, 'get_q_values') and is_deep_rl:
                # Deep RL methods with Q-networks
                try:
                    q_vals_array = agent.get_q_values(state)
                    q_values = {i: float(q_vals_array[i]) for i in range(len(q_vals_array))}
                except:
                    q_values = None
            elif hasattr(agent, 'get_action_probabilities'):
                # Policy gradient methods (REINFORCE)
                try:
                    action_probs = agent.get_action_probabilities(state)
                    q_values = {i: float(action_probs[i]) for i in range(len(action_probs))}
                except:
                    q_values = None
            
            # Store frame data
            frame_data = {
                'episode': episode_num,
                'step': step,
                'state': state.copy(),
                'action': action,
                'q_values': q_values,
                'env_name': self.env_name
            }
            
            # Take action
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            frame_data['reward'] = reward
            frame_data['next_state'] = next_state.copy()
            frame_data['terminated'] = terminated
            frame_data['truncated'] = truncated
            
            episode_data.append(frame_data)
            
            if terminated or truncated:
                break
            
            state = next_state
        
        return episode_data
    
    def visualize_frame(self, frame_idx):
        """Visualize a specific frame"""
        if not self.animation_data or frame_idx >= len(self.animation_data):
            return
        
        frame_data = self.animation_data[frame_idx]
        self.figure.clear()
        
        if self.env_name == "CartPole-v1":
            self.visualize_cartpole(frame_data)
        elif self.env_name == "MountainCar-v0":
            self.visualize_mountaincar(frame_data)
        elif self.env_name == "Acrobot-v1":
            self.visualize_acrobot(frame_data)
        elif self.env_name == "GridWorldTreasure-v0":
            self.visualize_gridworld_treasure(frame_data)
        else:
            self.visualize_generic(frame_data)
        
        self.canvas.draw()
    
    def visualize_cartpole(self, frame_data):
        """Visualize CartPole environment"""
        ax = self.figure.add_subplot(111)
        
        state = frame_data['state']
        cart_pos, cart_vel, pole_angle, pole_vel = state
        
        # Cart
        cart_width = 0.5
        cart_height = 0.3
        cart_x = cart_pos
        cart_y = 0
        
        cart = Rectangle((cart_x - cart_width/2, cart_y - cart_height/2), 
                        cart_width, cart_height, facecolor='blue', alpha=0.7)
        ax.add_patch(cart)
        
        # Pole
        pole_length = 1.0
        pole_x = cart_x + pole_length * np.sin(pole_angle)
        pole_y = cart_y + pole_length * np.cos(pole_angle)
        
        ax.plot([cart_x, pole_x], [cart_y, pole_y], 'r-', linewidth=5)
        ax.plot(pole_x, pole_y, 'ro', markersize=8)
        
        # Ground
        ax.axhline(y=-cart_height/2, color='brown', linewidth=3)
        
        # Set limits and labels
        ax.set_xlim(-3, 3)
        ax.set_ylim(-1, 2)
        ax.set_aspect('equal')
        ax.set_title(f"CartPole - Episode {frame_data['episode']}, Step {frame_data['step']}")
        
        # Add state information
        if self.show_state_info_cb.isChecked():
            info_text = f"Cart Pos: {cart_pos:.2f}\nCart Vel: {cart_vel:.2f}\n"
            info_text += f"Pole Angle: {pole_angle:.2f}\nPole Vel: {pole_vel:.2f}\n"
            info_text += f"Action: {'Right' if frame_data['action'] == 1 else 'Left'}\n"
            info_text += f"Reward: {frame_data['reward']:.1f}"
            
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add Q-values or action probabilities
        if self.show_qvalues_cb.isChecked() and frame_data['q_values']:
            # Determine if these are Q-values or action probabilities
            values = frame_data['q_values']
            max_val = max(values.values()) if values else 0
            
            if max_val <= 1.0 and min(values.values()) >= 0:
                # Likely action probabilities (REINFORCE)
                q_text = "Action Probabilities:\n"
                for action, prob in values.items():
                    action_name = "Right" if action == 1 else "Left"
                    q_text += f"{action_name}: {prob:.3f}\n"
            else:
                # Q-values (DQN-based methods)
                q_text = "Q-Values:\n"
                for action, q_val in values.items():
                    action_name = "Right" if action == 1 else "Left"
                    q_text += f"{action_name}: {q_val:.3f}\n"
            
            ax.text(0.98, 0.98, q_text, transform=ax.transAxes, 
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    def visualize_mountaincar(self, frame_data):
        """Visualize MountainCar environment"""
        ax = self.figure.add_subplot(111)
        
        state = frame_data['state']
        position, velocity = state
        
        # Mountain curve
        x = np.linspace(-1.2, 0.6, 100)
        y = np.sin(3 * x) * 0.45 + 0.55
        ax.plot(x, y, 'brown', linewidth=3)
        
        # Car
        car_x = position
        car_y = np.sin(3 * position) * 0.45 + 0.55 + 0.1
        
        car = Circle((car_x, car_y), 0.05, facecolor='red', alpha=0.8)
        ax.add_patch(car)
        
        # Velocity arrow
        if abs(velocity) > 0.001:
            arrow_scale = 0.3
            ax.arrow(car_x, car_y, velocity * arrow_scale, 0, 
                    head_width=0.03, head_length=0.02, fc='blue', ec='blue')
        
        # Goal flag
        goal_x = 0.5
        goal_y = np.sin(3 * goal_x) * 0.45 + 0.55 + 0.2
        ax.plot([goal_x, goal_x], [goal_y - 0.1, goal_y], 'g-', linewidth=3)
        ax.plot([goal_x, goal_x + 0.1], [goal_y, goal_y - 0.05], 'g-', linewidth=2)
        
        ax.set_xlim(-1.3, 0.7)
        ax.set_ylim(0, 1.5)
        ax.set_aspect('equal')
        ax.set_title(f"MountainCar - Episode {frame_data['episode']}, Step {frame_data['step']}")
        
        # Add state information
        if self.show_state_info_cb.isChecked():
            info_text = f"Position: {position:.3f}\nVelocity: {velocity:.3f}\n"
            actions = ["Left", "None", "Right"]
            info_text += f"Action: {actions[frame_data['action']]}\n"
            info_text += f"Reward: {frame_data['reward']:.1f}"
            
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add Q-values or action probabilities
        if self.show_qvalues_cb.isChecked() and frame_data['q_values']:
            # Determine if these are Q-values or action probabilities
            values = frame_data['q_values']
            max_val = max(values.values()) if values else 0
            
            if max_val <= 1.0 and min(values.values()) >= 0:
                # Likely action probabilities (REINFORCE)
                q_text = "Action Probabilities:\n"
                for action, prob in values.items():
                    action_name = ["Left", "None", "Right"][action]
                    q_text += f"{action_name}: {prob:.3f}\n"
            else:
                # Q-values (DQN-based methods)
                q_text = "Q-Values:\n"
                for action, q_val in values.items():
                    action_name = ["Left", "None", "Right"][action]
                    q_text += f"{action_name}: {q_val:.3f}\n"
            
            ax.text(0.98, 0.98, q_text, transform=ax.transAxes, 
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    def visualize_acrobot(self, frame_data):
        """Visualize Acrobot environment"""
        ax = self.figure.add_subplot(111)
        
        state = frame_data['state']
        cos_theta1, sin_theta1, cos_theta2, sin_theta2, theta1_dot, theta2_dot = state
        
        # Calculate angles
        theta1 = np.arctan2(sin_theta1, cos_theta1)
        theta2 = np.arctan2(sin_theta2, cos_theta2)
        
        # Link lengths
        l1, l2 = 1.0, 1.0
        
        # Joint positions
        x1 = l1 * sin_theta1
        y1 = -l1 * cos_theta1
        
        x2 = x1 + l2 * np.sin(theta1 + theta2)
        y2 = y1 - l2 * np.cos(theta1 + theta2)
        
        # Draw links
        ax.plot([0, x1], [0, y1], 'b-', linewidth=8, label='Link 1')
        ax.plot([x1, x2], [y1, y2], 'r-', linewidth=8, label='Link 2')
        
        # Draw joints
        ax.plot(0, 0, 'ko', markersize=10)  # Fixed joint
        ax.plot(x1, y1, 'go', markersize=8)  # Middle joint
        ax.plot(x2, y2, 'ro', markersize=8)  # End effector
        
        # Goal line (when end effector reaches height of first link)
        ax.axhline(y=-l1, color='green', linestyle='--', alpha=0.5, label='Goal')
        
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect('equal')
        ax.legend()
        ax.set_title(f"Acrobot - Episode {frame_data['episode']}, Step {frame_data['step']}")
        
        # Add state information
        if self.show_state_info_cb.isChecked():
            info_text = f"θ₁: {theta1:.2f} rad\nθ₂: {theta2:.2f} rad\n"
            info_text += f"θ₁̇: {theta1_dot:.2f}\nθ₂̇: {theta2_dot:.2f}\n"
            actions = ["CCW", "None", "CW"]
            info_text += f"Action: {actions[frame_data['action']]}\n"
            info_text += f"Reward: {frame_data['reward']:.1f}"
            
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add Q-values or action probabilities
        if self.show_qvalues_cb.isChecked() and frame_data['q_values']:
            # Determine if these are Q-values or action probabilities
            values = frame_data['q_values']
            max_val = max(values.values()) if values else 0
            
            if max_val <= 1.0 and min(values.values()) >= 0:
                # Likely action probabilities (REINFORCE)
                q_text = "Action Probabilities:\n"
                for action, prob in values.items():
                    action_name = ["CCW", "None", "CW"][action]
                    q_text += f"{action_name}: {prob:.3f}\n"
            else:
                # Q-values (DQN-based methods)
                q_text = "Q-Values:\n"
                for action, q_val in values.items():
                    action_name = ["CCW", "None", "CW"][action]
                    q_text += f"{action_name}: {q_val:.3f}\n"
            
            ax.text(0.98, 0.98, q_text, transform=ax.transAxes, 
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    def visualize_gridworld_treasure(self, frame_data):
        """Visualize GridWorld Treasure Hunt environment"""
        ax = self.figure.add_subplot(111)
        
        # Extract state information
        state = frame_data['state']
        
        # For GridWorldTreasure, we need to reconstruct the grid from the observation
        # The observation contains: [agent_x, agent_y, energy, steps_remaining, treasure_grid_flat, obstacle_grid_flat]
        agent_x, agent_y = int(state[0]), int(state[1])
        energy = int(state[2])
        steps_remaining = int(state[3])
        
        # Determine grid size (we need to infer this from the observation length)
        remaining_obs = len(state) - 4
        grid_size = int(np.sqrt(remaining_obs / 2))  # Divide by 2 for treasure and obstacle grids
        
        # Extract treasure and obstacle grids
        treasure_grid_flat = state[4:4 + grid_size**2]
        obstacle_grid_flat = state[4 + grid_size**2:]
        
        treasure_grid = treasure_grid_flat.reshape(grid_size, grid_size)
        obstacle_grid = obstacle_grid_flat.reshape(grid_size, grid_size)
        
        # Create display grid
        display_grid = np.zeros((grid_size, grid_size))
        
        # Add obstacles (1)
        display_grid[obstacle_grid.astype(bool)] = 1
        
        # Add treasures (2)
        display_grid[treasure_grid.astype(bool)] = 2
        
        # Add agent (3)
        display_grid[agent_x, agent_y] = 3
        
        # Create color map
        colors = ['white', 'brown', 'gold', 'blue']
        cmap = ListedColormap(colors)
        
        # Display grid
        im = ax.imshow(display_grid, cmap=cmap, vmin=0, vmax=3)
        
        # Add grid lines
        for i in range(grid_size + 1):
            ax.axhline(i - 0.5, color='black', linewidth=1)
            ax.axvline(i - 0.5, color='black', linewidth=1)
        
        # Set title and labels
        ax.set_title(f"GridWorld Treasure Hunt - Episode {frame_data['episode']}, Step {frame_data['step']}")
        ax.set_xticks(range(grid_size))
        ax.set_yticks(range(grid_size))
        ax.set_aspect('equal')
        
        # Add state information
        if self.show_state_info_cb.isChecked():
            info_text = f"Position: ({agent_x}, {agent_y})\n"
            info_text += f"Energy: {energy}\n"
            info_text += f"Steps Remaining: {steps_remaining}\n"
            actions = ["Up", "Right", "Down", "Left", "Stay"]
            info_text += f"Action: {actions[frame_data['action']]}\n"
            info_text += f"Reward: {frame_data['reward']:.1f}"
            
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add Q-values or action probabilities
        if self.show_qvalues_cb.isChecked() and frame_data['q_values']:
            # Determine if these are Q-values or action probabilities
            values = frame_data['q_values']
            max_val = max(values.values()) if values else 0
            
            if max_val <= 1.0 and min(values.values()) >= 0:
                # Likely action probabilities (REINFORCE)
                q_text = "Action Probabilities:\n"
                for action, prob in values.items():
                    action_name = ["Up", "Right", "Down", "Left", "Stay"][action]
                    q_text += f"{action_name}: {prob:.3f}\n"
            else:
                # Q-values (DQN-based methods)
                q_text = "Q-Values:\n"
                for action, q_val in values.items():
                    action_name = ["Up", "Right", "Down", "Left", "Stay"][action]
                    q_text += f"{action_name}: {q_val:.3f}\n"
            
            ax.text(0.98, 0.98, q_text, transform=ax.transAxes, 
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Add legend
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='black', label='Empty'),
            plt.Rectangle((0, 0), 1, 1, facecolor='brown', label='Obstacle'),
            plt.Rectangle((0, 0), 1, 1, facecolor='gold', label='Treasure'),
            plt.Rectangle((0, 0), 1, 1, facecolor='blue', label='Agent')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    def visualize_generic(self, frame_data):
        """Generic visualization for unknown environments"""
        ax = self.figure.add_subplot(111)
        
        state = frame_data['state']
        
        # Simple bar chart of state values
        ax.bar(range(len(state)), state)
        ax.set_xlabel('State Dimension')
        ax.set_ylabel('Value')
        ax.set_title(f"State Visualization - Episode {frame_data['episode']}, Step {frame_data['step']}")
        
        # Add state information
        if self.show_state_info_cb.isChecked():
            info_text = f"State: {state}\n"
            info_text += f"Action: {frame_data['action']}\n"
            info_text += f"Reward: {frame_data['reward']:.1f}"
            
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def toggle_playback(self):
        """Toggle animation playback"""
        if self.is_playing:
            self.timer.stop()
            self.play_button.setText("Play")
            self.is_playing = False
        else:
            self.timer.start(100)  # 10 FPS base rate
            self.play_button.setText("Pause")
            self.is_playing = True
    
    def stop_playback(self):
        """Stop animation playback"""
        self.timer.stop()
        self.play_button.setText("Play")
        self.is_playing = False
        self.current_frame = 0
        self.frame_slider.setValue(0)
        self.visualize_frame(0)
    
    def update_frame(self):
        """Update to next frame in animation"""
        if self.current_frame < len(self.animation_data) - 1:
            self.current_frame += 1
        else:
            self.current_frame = 0  # Loop animation
        
        self.frame_slider.setValue(self.current_frame)
        self.visualize_frame(self.current_frame)
        self.frame_label.setText(f"{self.current_frame} / {len(self.animation_data)}")
    
    def seek_frame(self, frame_idx):
        """Seek to specific frame"""
        self.current_frame = frame_idx
        self.visualize_frame(frame_idx)
        self.frame_label.setText(f"{frame_idx} / {len(self.animation_data)}")
    
    def update_animation_speed(self, speed):
        """Update animation speed"""
        self.speed_label.setText(f"{speed}x")
        if self.is_playing:
            # Adjust timer interval (lower = faster)
            interval = max(10, 200 // speed)
            self.timer.setInterval(interval)
    
    def save_animation(self):
        """Save animation data to file"""
        if not self.animation_data:
            QMessageBox.warning(self, "Warning", "No animation data to save!")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Animation", "", "Animation Files (*.pkl);;All Files (*)"
        )
        
        if filename:
            try:
                with open(filename, 'wb') as f:
                    pickle.dump({
                        'animation_data': self.animation_data,
                        'env_name': self.env_name,
                        'metadata': {
                            'total_frames': len(self.animation_data),
                            'episodes': max(frame['episode'] for frame in self.animation_data) + 1
                        }
                    }, f)
                QMessageBox.information(self, "Success", f"Animation saved to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save animation: {str(e)}")
    
    def load_animation(self):
        """Load animation data from file"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Animation", "", "Animation Files (*.pkl);;All Files (*)"
        )
        
        if filename:
            try:
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
                
                self.animation_data = data['animation_data']
                self.env_name = data['env_name']
                
                # Enable playback controls
                self.play_button.setEnabled(True)
                self.stop_button.setEnabled(True)
                self.frame_slider.setEnabled(True)
                self.save_button.setEnabled(True)
                
                self.frame_slider.setRange(0, len(self.animation_data) - 1)
                self.frame_label.setText(f"0 / {len(self.animation_data)}")
                
                metadata = data.get('metadata', {})
                episodes = metadata.get('episodes', 'unknown')
                self.status_label.setText(f"Loaded {len(self.animation_data)} frames from {episodes} episodes")
                
                # Show first frame
                self.current_frame = 0
                self.visualize_frame(0)
                
                QMessageBox.information(self, "Success", f"Animation loaded from {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load animation: {str(e)}")
    
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
            # GridWorldTreasure - only discretize key features for tabular methods
            grid_size = getattr(env, 'grid_size', 8)
            max_energy = getattr(env, 'max_energy', 100)
            max_steps = getattr(env, 'max_steps', 200)
            
            state_bounds = [
                (0, grid_size - 1),     # agent_x
                (0, grid_size - 1),     # agent_y  
                (0, max_energy),        # energy
                (0, max_steps)          # steps_remaining
            ]
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
        return tuple(np.digitize(observation[i], bins[i]) for i in range(len(observation))) 