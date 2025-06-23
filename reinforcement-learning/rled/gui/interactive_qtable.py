"""
Interactive Q-table visualization with dimension control and multiple visualization modes
"""
import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, 
                            QLabel, QSlider, QCheckBox, QComboBox, QPushButton,
                            QGroupBox, QSpinBox)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class InteractiveQTableWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.q_tables = {}
        self.env_name = ""
        self.state_bounds = []
        self.discretization_bins = 10
        
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the interactive visualization UI"""
        layout = QVBoxLayout(self)
        
        # Control panel
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel)
        
        # Visualization area
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
    def create_control_panel(self):
        """Create the control panel with dimension controls and visualization options"""
        control_widget = QWidget()
        layout = QVBoxLayout(control_widget)
        
        # Title
        title = QLabel("Interactive Q-Table Visualization")
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(title)
        
        # Main controls layout
        main_controls = QHBoxLayout()
        
        # Dimension controls
        self.dimension_group = self.create_dimension_controls()
        main_controls.addWidget(self.dimension_group)
        
        # Visualization options
        viz_group = self.create_visualization_options()
        main_controls.addWidget(viz_group)
        
        # Algorithm selection
        algo_group = self.create_algorithm_selection()
        main_controls.addWidget(algo_group)
        
        layout.addLayout(main_controls)
        
        # Update button
        self.update_button = QPushButton("Update Visualization")
        self.update_button.clicked.connect(self.update_visualization)
        layout.addWidget(self.update_button)
        
        return control_widget
    
    def create_dimension_controls(self):
        """Create dimension control widgets"""
        group = QGroupBox("Dimension Controls")
        # Don't set layout here - it will be set when we create controls
        
        # Will be populated when Q-table data is available
        self.dimension_controls = {}
        
        return group
    
    def create_visualization_options(self):
        """Create visualization option controls"""
        group = QGroupBox("Visualization Options")
        layout = QVBoxLayout(group)
        
        # Visualization mode
        layout.addWidget(QLabel("Visualization Mode:"))
        self.viz_mode_combo = QComboBox()
        modes = [
            "Q-Value Heatmap",
            "Policy (Argmax)",
            "Q-Value Difference",
            "Value Function"
        ]
        if SKLEARN_AVAILABLE:
            modes.append("PCA Projection")
        self.viz_mode_combo.addItems(modes)
        layout.addWidget(self.viz_mode_combo)
        
        # Color scheme
        layout.addWidget(QLabel("Color Scheme:"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems([
            "viridis", "plasma", "inferno", "magma",
            "coolwarm", "RdYlBu", "seismic"
        ])
        layout.addWidget(self.colormap_combo)
        
        # PCA components (only for PCA mode)
        if SKLEARN_AVAILABLE:
            layout.addWidget(QLabel("PCA Components:"))
            self.pca_components_spin = QSpinBox()
            self.pca_components_spin.setRange(2, 4)
            self.pca_components_spin.setValue(2)
            layout.addWidget(self.pca_components_spin)
        
        return group
    
    def create_algorithm_selection(self):
        """Create algorithm selection controls"""
        group = QGroupBox("Algorithm")
        layout = QVBoxLayout(group)
        
        self.algorithm_combo = QComboBox()
        layout.addWidget(self.algorithm_combo)
        
        return group
    
    def update_q_tables(self, q_tables, env_name, state_bounds, discretization_bins):
        """Update the Q-tables and environment information"""
        self.q_tables = q_tables
        self.env_name = env_name
        self.state_bounds = state_bounds
        self.discretization_bins = discretization_bins
        
        # Update algorithm combo
        self.algorithm_combo.clear()
        self.algorithm_combo.addItems(list(q_tables.keys()))
        
        # Create dimension controls based on environment
        self.create_dimension_controls_for_env()
        
        # Initial visualization
        self.update_visualization()
    
    def create_dimension_controls_for_env(self):
        """Create dimension controls based on the environment"""
        # Clear existing controls and layout
        if self.dimension_group.layout():
            # Remove all widgets from the existing layout
            while self.dimension_group.layout().count():
                child = self.dimension_group.layout().takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
            # Delete the old layout
            self.dimension_group.layout().deleteLater()
        
        # Create new layout
        layout = QGridLayout()
        self.dimension_group.setLayout(layout)
        self.dimension_controls = {}
        
        # Get dimension names based on environment
        dim_names = self.get_dimension_names()
        
        for i, dim_name in enumerate(dim_names):
            # Checkbox to freeze/unfreeze dimension
            checkbox = QCheckBox(f"Freeze {dim_name}")
            layout.addWidget(checkbox, i, 0)
            
            # Slider for frozen value
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, self.discretization_bins - 1)
            slider.setValue(self.discretization_bins // 2)
            slider.setEnabled(False)
            layout.addWidget(slider, i, 1)
            
            # Value label
            value_label = QLabel(str(slider.value()))
            layout.addWidget(value_label, i, 2)
            
            # Connect signals
            checkbox.toggled.connect(lambda checked, s=slider: s.setEnabled(checked))
            checkbox.toggled.connect(self.update_visualization)
            slider.valueChanged.connect(lambda val, lbl=value_label: lbl.setText(str(val)))
            slider.valueChanged.connect(self.update_visualization)
            
            self.dimension_controls[i] = {
                'name': dim_name,
                'checkbox': checkbox,
                'slider': slider,
                'label': value_label
            }
    
    def get_dimension_names(self):
        """Get dimension names based on environment"""
        if self.env_name == "CartPole-v1":
            return ["Cart Position", "Cart Velocity", "Pole Angle", "Pole Angular Velocity"]
        elif self.env_name == "MountainCar-v0":
            return ["Position", "Velocity"]
        elif self.env_name == "Acrobot-v1":
            return ["cos(θ₁)", "sin(θ₁)", "cos(θ₂)", "sin(θ₂)", "θ₁̇", "θ₂̇"]
        else:
            n_dims = len(self.state_bounds) if self.state_bounds else 4
            return [f"Dimension {i+1}" for i in range(n_dims)]
    
    def update_visualization(self):
        """Update the visualization based on current settings"""
        if not self.q_tables:
            return
        
        self.figure.clear()
        
        viz_mode = self.viz_mode_combo.currentText()
        
        if viz_mode == "PCA Projection" and SKLEARN_AVAILABLE:
            self.create_pca_visualization()
        else:
            self.create_2d_visualization(viz_mode)
        
        self.canvas.draw()
    
    def create_2d_visualization(self, viz_mode):
        """Create 2D visualization with frozen dimensions"""
        # Get active (non-frozen) dimensions
        active_dims = []
        frozen_values = {}
        
        for dim_idx, controls in self.dimension_controls.items():
            if controls['checkbox'].isChecked():
                frozen_values[dim_idx] = controls['slider'].value()
            else:
                active_dims.append(dim_idx)
        
        if len(active_dims) < 2:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "Please unfreeze at least 2 dimensions for 2D visualization",
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Use first two active dimensions for visualization
        x_dim, y_dim = active_dims[0], active_dims[1]
        
        # Get current algorithm
        algo_name = self.algorithm_combo.currentText()
        if algo_name not in self.q_tables:
            return
        
        q_table = self.q_tables[algo_name]
        
        # Create grid for visualization
        x_range = range(self.discretization_bins)
        y_range = range(self.discretization_bins)
        
        if viz_mode == "Q-Value Heatmap":
            self.create_q_value_heatmap(q_table, x_dim, y_dim, frozen_values, x_range, y_range)
        elif viz_mode == "Policy (Argmax)":
            self.create_policy_visualization(q_table, x_dim, y_dim, frozen_values, x_range, y_range)
        elif viz_mode == "Q-Value Difference":
            self.create_q_difference_visualization(x_dim, y_dim, frozen_values, x_range, y_range)
        elif viz_mode == "Value Function":
            self.create_value_function_visualization(q_table, x_dim, y_dim, frozen_values, x_range, y_range)
    
    def create_q_value_heatmap(self, q_table, x_dim, y_dim, frozen_values, x_range, y_range):
        """Create Q-value heatmap for specific action"""
        n_actions = len(list(q_table.values())[0]) if q_table else 2
        
        # Create subplots for each action
        for action in range(n_actions):
            ax = self.figure.add_subplot(1, n_actions, action + 1)
            
            # Create data matrix
            data = np.zeros((len(y_range), len(x_range)))
            
            for i, y_val in enumerate(y_range):
                for j, x_val in enumerate(x_range):
                    state = self.construct_state(x_dim, y_dim, x_val, y_val, frozen_values)
                    
                    if state in q_table:
                        data[i, j] = q_table[state][action]
            
            # Plot heatmap
            im = ax.imshow(data, cmap=self.colormap_combo.currentText(), 
                          aspect='auto', origin='lower')
            
            # Labels and title
            dim_names = self.get_dimension_names()
            ax.set_xlabel(f"{dim_names[x_dim]} (Dim {x_dim})")
            ax.set_ylabel(f"{dim_names[y_dim]} (Dim {y_dim})")
            
            action_labels = self.get_action_labels()
            action_name = action_labels[action] if action < len(action_labels) else f"Action {action}"
            
            # Add frozen dimensions info to title
            frozen_info = ""
            if frozen_values:
                frozen_dims = [f"{dim_names[dim]}={val}" for dim, val in frozen_values.items()]
                frozen_info = f" (Frozen: {', '.join(frozen_dims)})"
            
            ax.set_title(f'{action_name} Q-Values{frozen_info}')
            
            # Colorbar
            self.figure.colorbar(im, ax=ax)
    
    def create_policy_visualization(self, q_table, x_dim, y_dim, frozen_values, x_range, y_range):
        """Create policy (argmax) visualization"""
        ax = self.figure.add_subplot(111)
        
        # Create data matrix for policy
        policy_data = np.zeros((len(y_range), len(x_range)))
        
        for i, y_val in enumerate(y_range):
            for j, x_val in enumerate(x_range):
                state = self.construct_state(x_dim, y_dim, x_val, y_val, frozen_values)
                
                if state in q_table:
                    best_action = max(q_table[state], key=q_table[state].get)
                    policy_data[i, j] = best_action
        
        # Plot policy
        im = ax.imshow(policy_data, cmap='tab10', aspect='auto', origin='lower')
        
        # Labels and title
        dim_names = self.get_dimension_names()
        ax.set_xlabel(f"{dim_names[x_dim]} (Dim {x_dim})")
        ax.set_ylabel(f"{dim_names[y_dim]} (Dim {y_dim})")
        
        # Add frozen dimensions info to title
        frozen_info = ""
        if frozen_values:
            frozen_dims = [f"{dim_names[dim]}={val}" for dim, val in frozen_values.items()]
            frozen_info = f" (Frozen: {', '.join(frozen_dims)})"
        
        ax.set_title(f'Learned Policy (Best Actions){frozen_info}')
        
        # Colorbar with action labels
        cbar = self.figure.colorbar(im, ax=ax)
        action_labels = self.get_action_labels()
        if len(action_labels) <= 10:
            cbar.set_ticks(range(len(action_labels)))
            cbar.set_ticklabels(action_labels)
    
    def create_q_difference_visualization(self, x_dim, y_dim, frozen_values, x_range, y_range):
        """Create Q-value difference visualization between algorithms"""
        if len(self.q_tables) < 2:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "Need at least 2 algorithms for difference visualization",
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        algos = list(self.q_tables.keys())
        q_table1, q_table2 = self.q_tables[algos[0]], self.q_tables[algos[1]]
        
        n_actions = len(list(q_table1.values())[0]) if q_table1 else 2
        
        for action in range(n_actions):
            ax = self.figure.add_subplot(1, n_actions, action + 1)
            
            diff_data = np.zeros((len(y_range), len(x_range)))
            
            for i, y_val in enumerate(y_range):
                for j, x_val in enumerate(x_range):
                    state = self.construct_state(x_dim, y_dim, x_val, y_val, frozen_values)
                    
                    q1 = q_table1.get(state, {}).get(action, 0)
                    q2 = q_table2.get(state, {}).get(action, 0)
                    diff_data[i, j] = q1 - q2
            
            im = ax.imshow(diff_data, cmap='RdBu', aspect='auto', origin='lower')
            
            dim_names = self.get_dimension_names()
            ax.set_xlabel(dim_names[x_dim])
            ax.set_ylabel(dim_names[y_dim])
            ax.set_title(f'Q-Value Difference (Action {action})\n{algos[0]} - {algos[1]}')
            
            self.figure.colorbar(im, ax=ax)
    
    def create_value_function_visualization(self, q_table, x_dim, y_dim, frozen_values, x_range, y_range):
        """Create value function (max Q-value) visualization"""
        ax = self.figure.add_subplot(111)
        
        value_data = np.zeros((len(y_range), len(x_range)))
        
        for i, y_val in enumerate(y_range):
            for j, x_val in enumerate(x_range):
                state = self.construct_state(x_dim, y_dim, x_val, y_val, frozen_values)
                
                if state in q_table:
                    value_data[i, j] = max(q_table[state].values())
        
        im = ax.imshow(value_data, cmap=self.colormap_combo.currentText(), 
                      aspect='auto', origin='lower')
        
        dim_names = self.get_dimension_names()
        ax.set_xlabel(dim_names[x_dim])
        ax.set_ylabel(dim_names[y_dim])
        ax.set_title('Value Function (Max Q-Values)')
        
        self.figure.colorbar(im, ax=ax, label='Value')
    
    def create_pca_visualization(self):
        """Create PCA projection visualization"""
        if not SKLEARN_AVAILABLE:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "sklearn not available for PCA visualization",
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        if not self.q_tables:
            return
        
        all_states = set()
        for q_table in self.q_tables.values():
            all_states.update(q_table.keys())
        
        if len(all_states) < 3:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "Need more states for PCA visualization",
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        states_list = list(all_states)
        X = np.array([[float(dim) for dim in state] for state in states_list])
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        n_components = min(self.pca_components_spin.value(), X_scaled.shape[1])
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        algo_name = self.algorithm_combo.currentText()
        if algo_name in self.q_tables:
            q_table = self.q_tables[algo_name]
            values = [max(q_table.get(state, {0: 0}).values()) for state in states_list]
        else:
            values = [0] * len(states_list)
        
        if n_components >= 2:
            ax = self.figure.add_subplot(111)
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=values, 
                               cmap=self.colormap_combo.currentText())
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            ax.set_title('PCA Projection of State Space')
            self.figure.colorbar(scatter, ax=ax, label='Max Q-Value')
            
            total_var = sum(pca.explained_variance_ratio_)
            ax.text(0.02, 0.98, f'Total variance explained: {total_var:.2%}',
                   transform=ax.transAxes, va='top')
    
    def construct_state(self, x_dim, y_dim, x_val, y_val, frozen_values):
        """Construct state tuple with frozen and active dimensions"""
        n_dims = len(self.get_dimension_names())
        state = [0] * n_dims
        
        for dim, val in frozen_values.items():
            state[dim] = val
        
        state[x_dim] = x_val
        state[y_dim] = y_val
        
        return tuple(state)
    
    def get_action_labels(self):
        """Get action labels based on environment"""
        if self.env_name == "CartPole-v1":
            return ["Left", "Right"]
        elif self.env_name == "MountainCar-v0":
            return ["Left", "None", "Right"]
        elif self.env_name == "Acrobot-v1":
            return ["CCW", "None", "CW"]
        else:
            return ["Action 0", "Action 1"] 