# RLEd - Reinforcement Learning Education Tool

RLEd is an open-source, low-code education tool designed to help visualize and teach reinforcement learning concepts. Built on top of OpenAI's Gymnasium and PyTorch, it provides an intuitive GUI interface for understanding both tabular and deep RL algorithms through interactive visualizations and real-time training analysis.

## Features

### Algorithms Supported
**Tabular Methods:**
- Q-Learning
- SARSA  
- Expected SARSA
- Double Q-Learning
- Monte Carlo Control
- n-step SARSA

**Deep RL Methods:**
- Deep Q-Network (DQN)
- Double DQN
- Dueling DQN
- REINFORCE (Policy Gradient)

### Environments Supported
**Built-in Gymnasium Environments:**
- CartPole-v1 (Classic control - pole balancing)
- MountainCar-v0 (Continuous control - hill climbing)
- Acrobot-v1 (Complex dynamics - pendulum swing-up)

**Custom Educational Environments:**
- GridWorldTreasure-v0 (Resource management and exploration)

### Visualization Features
- **Real-time Training Progress:** Live plots with moving averages and raw data
- **Interactive Q-table Visualization:** 
  - Multi-dimensional state space exploration
  - Dimension freezing and slicing
  - Multiple visualization modes (Q-values, policy, value function)
  - PCA dimensionality reduction for high-dimensional spaces
- **Policy Animation:** 
  - Physical system visualization for all environments
  - Step-by-step policy execution with Q-values/action probabilities
  - Customizable playback speed and frame navigation
- **Algorithm Comparison:** Side-by-side performance analysis
- **Comprehensive Logging:** Detailed training statistics and results

## Requirements

- **Python 3.8+**
- **PyQt6** (for GUI interface)
- **PyTorch** (for deep RL algorithms)
- **Gymnasium** (for RL environments)
- **NumPy, Matplotlib, Pandas** (for data processing and visualization)
- **scikit-learn** (optional, for PCA in Q-table visualization)

## Installation

### Quick Start
```bash
# Clone the repository
git clone https://github.com/yourusername/RLEd.git
cd RLEd

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python rled_app.py
```

### Advanced Installation Options
```bash
# Option 1: Auto-detect CUDA (recommended)
pip install -r requirements.txt

# Option 2: Specific CUDA version
pip install -r requirements-cuda118.txt  # For CUDA 11.8
pip install -r requirements-cuda121.txt  # For CUDA 12.1

# Option 3: CPU-only (no GPU acceleration)
pip install -r requirements-cpu.txt

# Option 4: Development installation
pip install -e .
```

## Usage

### GUI Application
```bash
# Run the main RLEd application
python rled_app.py
```

### Command Line Examples
```bash
# Run the GridWorld Treasure Hunt example
python examples/gridworld_treasure_example.py

# Test basic Q-Learning on CartPole
python examples/basic_example.py
```

### Key Features Overview
1. **Algorithm Selection**: Choose from 10+ different RL algorithms (tabular and deep)
2. **Environment Configuration**: 4 environments with customizable parameters
3. **Real-time Training**: Watch algorithms learn with live progress visualization
4. **Interactive Q-table Analysis**: Explore learned policies in multi-dimensional space
5. **Policy Animation**: See trained agents perform with step-by-step breakdowns
6. **Algorithm Comparison**: Compare multiple algorithms side-by-side
7. **Educational Focus**: Designed specifically for learning and teaching RL concepts

### Example Workflows

**For Beginners:**
1. Select "Q-Learning" algorithm
2. Choose "CartPole-v1" environment  
3. Use default parameters
4. Click "Start Training" and watch the learning curve
5. Explore the Q-table visualization to understand state-action values
6. Use Policy Animation to see the trained agent in action

**For Advanced Users:**
1. Select "Compare All" to run multiple algorithms
2. Choose "GridWorldTreasure-v0" for complex multi-objective learning
3. Adjust hyperparameters (learning rate, epsilon, network architecture)
4. Analyze performance differences between tabular and deep methods
5. Use interactive Q-table visualization with PCA for high-dimensional analysis

**For Educators:**
1. Use the tool to demonstrate different RL concepts in real-time
2. Show the difference between exploration vs exploitation with epsilon settings
3. Compare tabular methods (Q-Learning) vs deep methods (DQN) on the same task
4. Use GridWorld environment to teach resource management and multi-objective RL

## Custom Environment: GridWorld Treasure Hunt

RLEd includes a custom educational environment designed to teach advanced RL concepts:

### Environment Features
- **Grid-based world** with treasures, obstacles, and energy management
- **Multi-objective rewards**: treasure collection, energy efficiency, time optimization
- **Resource management**: Limited energy that depletes with movement and recovers when resting
- **Strategic decision making**: When to move vs when to rest
- **5 actions**: Up, Right, Down, Left, Stay (to recover energy)

### Educational Value
- **Sparse rewards**: Teaches exploration in environments with infrequent positive feedback
- **Multi-objective optimization**: Balancing multiple competing objectives
- **Resource constraints**: Planning under limited resources
- **State space complexity**: Demonstrates difference between tabular and deep RL approaches

### Reward Structure
- **+100** for collecting a treasure
- **-1** for each step (time penalty)
- **-5** for hitting obstacles  
- **-10** for running out of energy
- **+50** bonus for collecting all treasures

## Screenshots

*Coming soon - GUI screenshots showing the main interface, Q-table visualization, and policy animation.*

## Project Structure

```
RLEd/
├── rled/                          # Main package
│   ├── algorithms/                # RL algorithm implementations
│   │   ├── tabular/              # Q-Learning, SARSA, etc.
│   │   └── deep/                 # DQN, REINFORCE, etc.
│   ├── environments/             # Custom environments
│   ├── gui/                      # PyQt6 GUI components
│   └── visualization/            # Plotting and animation utilities
├── examples/                     # Example scripts and tutorials
├── requirements*.txt             # Dependency specifications
├── rled_app.py                  # Main application launcher
└── README.md                    # This file
```

## Contributing

We welcome contributions! Here are some ways you can help:

- **Bug Reports**: Found a bug? Please open an issue with details
- **Feature Requests**: Have an idea for a new feature?
- **Code Contributions**: Submit pull requests for bug fixes or new features
- **Documentation**: Help improve documentation and examples
- **Educational Content**: Create tutorials or example environments

### Development Setup
```bash
# Clone and install in development mode
git clone https://github.com/yourusername/RLEd.git
cd RLEd
pip install -e .

# Run tests (when available)
python -m pytest tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **OpenAI Gymnasium** for providing the foundation RL environments
- **PyTorch** team for the deep learning framework
- **PyQt6** for the powerful GUI framework
- **The RL community** for research and open-source contributions
- **Educators and students** who inspire better learning tools

## Citation

If you use RLEd in your research or educational work, please consider citing:

```bibtex
@software{rled2024,
  title={RLEd: An Open-Source Reinforcement Learning Education Tool},
  author={Jerry Ng},
  year={2025},
  url={https://github.com/jerrying123/RLEd}
}
``` 