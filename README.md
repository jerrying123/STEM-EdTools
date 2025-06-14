# RLEd - Reinforcement Learning Education Tool

RLEd is an open-source education tool designed to help visualize and teach reinforcement learning concepts. Built on top of OpenAI's Gymnasium and PyTorch, it provides an intuitive GUI interface for understanding both tabular and deep RL algorithms.

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

### Visualization Features
- Real-time training progress plots
- Interactive Q-table visualization with dimension control
- Policy animation with physical system visualization
- Comprehensive comparison between algorithms
- Support for CartPole, MountainCar, and Acrobot environments

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/RLEd.git
cd RLEd

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (choose one based on your system)

# Option 1: Auto-detect CUDA (recommended)
pip install -r requirements.txt

# Option 2: Specific CUDA version
pip install -r requirements-cuda118.txt  # For CUDA 11.8
pip install -r requirements-cuda121.txt  # For CUDA 12.1

# Option 3: CPU-only (no GPU acceleration)
pip install -r requirements-cpu.txt
```

## Usage

### GUI Application
```bash
# Run the main GUI application
python run_gui.py
```

### Features Overview
1. **Algorithm Selection**: Choose from 10 different RL algorithms
2. **Environment Configuration**: Select environment and adjust parameters
3. **Training Visualization**: Watch real-time training progress
4. **Interactive Analysis**: Explore Q-tables and policy decisions
5. **Policy Animation**: See trained agents in action

### Example Workflow
1. Select an algorithm (e.g., "DQN" for deep RL or "Q-Learning" for tabular)
2. Choose environment (CartPole-v1 recommended for beginners)
3. Adjust hyperparameters as needed
4. Click "Start Training" and watch the learning process
5. Explore results in the visualization tabs

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI Gymnasium for providing the RL environments
- The open-source community for inspiration and support 