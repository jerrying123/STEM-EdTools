# STEM-EdTools

A comprehensive collection of open-source educational tools for teaching and learning various STEM concepts, with a focus on machine learning and artificial intelligence.

## Overview

STEM-EdTools is a repository containing multiple educational applications designed to make complex STEM concepts accessible through interactive visualizations and hands-on learning experiences. Each tool is self-contained with its own documentation and can be used independently.

## Available Tools

### 🤖 [Reinforcement Learning Education Tool (RLEd)](reinforcement-learning/README.md)

An interactive GUI application for learning reinforcement learning concepts through visualization and real-time training analysis.

**Key Features:**
- 10+ RL algorithms (Q-Learning, DQN, REINFORCE, etc.)
- 4 environments (CartPole, MountainCar, Acrobot, GridWorld)
- Real-time training visualization
- Interactive Q-table exploration
- Policy animation and comparison tools

**Perfect for:** Students learning RL, educators teaching AI concepts, researchers exploring algorithm performance

### 📊 [Sampling Biases Learning Tool](sampling-biases/README.md)

An interactive application that teaches about sampling biases and data representativeness in machine learning using the MNIST dataset.

**Key Features:**
- Interactive class selection for digit datasets
- Dynamic dataset size control with train/val/test splits
- Real-time neural network training
- Performance analysis with per-class accuracy
- Educational experiments demonstrating various types of sampling bias
- Confusion matrix visualization with matplotlib heatmaps

**Perfect for:** Machine learning beginners, data science students, understanding bias in data collection and preparation

### 🎯 [ML Context Capture & Analysis Tool](ml-application/README.md)

An interactive application that demonstrates real-world AI integration by capturing screen content and analyzing it with OpenAI's GPT-4 Vision model.

**Key Features:**
- Transparent overlay window for screen capture
- Interactive area selection with real-time preview
- GPT-4 Vision integration for image analysis
- Contextual prompting with example suggestions
- Professional GUI with status feedback
- Educational demonstrations of AI capabilities

**Perfect for:** AI/ML demonstrations, computer vision education, practical AI integration examples

### 🎨 [GenAI Image Generation Tool](genai/README.md)

An interactive application for generating images using state-of-the-art AI models from Hugging Face, demonstrating the power of generative AI.

**Key Features:**
- Multiple AI models (Stable Diffusion, DALL-E Mini)
- Custom prompts with example suggestions
- Advanced generation controls (size, quality, steps)
- Interactive image gallery with batch operations
- Real-time generation logging and progress tracking
- Educational demonstrations of generative AI capabilities

**Perfect for:** Creative AI education, generative model exploration, understanding text-to-image generation

## Quick Start

### Prerequisites
- Python 3.8+
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/STEM-EdTools.git
   cd STEM-EdTools
   ```

2. **Set up a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install all dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your OpenAI API key (for ML Context Capture Tool):**
   Create a `.env` file in the project directory:
   ```
   OPENAI_KEY=your_openai_api_key_here
   ```

5. **Run any tool:**
   ```bash
   # Reinforcement Learning Tool
   cd reinforcement-learning
   python rled_app.py
   
   # Sampling Biases Tool
   cd ../sampling-biases
   python main.py
   
   # ML Context Capture Tool
   cd ../ml-application
   python main.py
   
   # GenAI Image Generation Tool
   cd ../genai
   python main.py
   ```

## Usage Guide

### For Students
1. **Start with Sampling Biases Tool** if you're new to machine learning and data science
2. **Progress to RLEd** once you understand basic ML concepts and bias awareness
3. **Try ML Context Capture Tool** to see AI in action with real-world applications
4. **Explore GenAI Tool** to understand generative AI and creative applications
5. **Experiment freely** - each tool is designed for learning through exploration

### For Educators
1. **Use Sampling Biases Tool** to demonstrate data bias, selection bias, and sampling bias concepts
2. **Use RLEd** to teach reinforcement learning algorithms and concepts
3. **Use ML Context Capture Tool** to show practical AI applications and computer vision
4. **Use GenAI Tool** to demonstrate generative AI capabilities and creative applications
5. **Customize parameters** to create specific learning scenarios
6. **Compare algorithms** to show different approaches to the same problem

### For Researchers
1. **Benchmark algorithms** using RLEd's comparison features
2. **Test bias hypotheses** about data composition effects with the sampling biases tool
3. **Demonstrate AI capabilities** using the ML context capture tool
4. **Explore generative models** using the GenAI tool for creative AI research
5. **Extend tools** for specific research needs

## Educational Pathways

### Beginner Path
1. **Sampling Biases Tool** → Understanding bias in ML and data science
2. **RLEd (Q-Learning)** → Basic RL concepts
3. **ML Context Capture Tool** → Practical AI applications
4. **GenAI Tool** → Generative AI and creative applications
5. **RLEd (DQN)** → Deep RL introduction

### Advanced Path
1. **RLEd (Policy Gradient)** → Advanced RL algorithms
2. **Custom environments** → Building your own RL problems
3. **Algorithm comparison** → Research-level analysis
4. **Advanced GenAI** → Custom models and fine-tuning
5. **AI integration** → Building custom AI applications

## Contributing

We welcome contributions to make these tools even better for education!

### How to Contribute
1. **Fork the repository**
2. **Create a feature branch** for your changes
3. **Make your improvements** (new algorithms, environments, visualizations)
4. **Test thoroughly** to ensure educational value
5. **Submit a pull request** with clear documentation

### Areas for Contribution
- **New algorithms** for RLEd
- **Additional bias types** for the sampling biases tool
- **AI model integrations** for the ML context capture tool
- **New generative models** for the GenAI tool
- **Improved visualizations** and UI enhancements
- **Educational content** and tutorials
- **Performance optimizations**
- **Documentation improvements**

## Project Structure

```
STEM-EdTools/
├── reinforcement-learning/     # RLEd - RL education tool
│   ├── rled/                  # Main package
│   ├── examples/              # Example scripts
│   ├── requirements*.txt      # Dependencies
│   └── README.md             # Detailed documentation
├── sampling-biases/           # Sampling biases learning tool
│   ├── data/                 # Dataset storage
│   ├── main.py              # Main application
│   ├── model.py             # Neural network model
│   ├── trainer.py           # Training logic
│   ├── data_handler.py      # Data processing
│   └── README.md            # Detailed documentation
├── ml-application/           # ML context capture tool
│   ├── main.py              # Main application
│   └── README.md            # Detailed documentation
├── genai/                   # GenAI image generation tool
│   ├── main.py              # Main application
│   └── README.md            # Detailed documentation
├── requirements.txt          # Root dependencies
├── LICENSE.md               # MIT License
└── README.md               # This file
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- **OpenAI** for GPT-4 Vision API
- **OpenAI Gymnasium** for RL environments
- **PyTorch** for deep learning framework
- **MNIST Dataset** creators
- **The open-source community** for inspiration and tools
- **Educators and students** who inspire better learning tools

## Citation

If you use STEM-EdTools in your research or educational work, please consider citing:

```bibtex
@software{stemedtools2024,
  title={STEM-EdTools: A Collection of Educational Tools for STEM Learning},
  author={Jerry Ng},
  year={2024},
  url={https://github.com/jerrying123/STEM-EdTools}
}
```

## Support

- **Issues**: Found a bug or have a feature request? [Open an issue](https://github.com/yourusername/STEM-EdTools/issues)
- **Discussions**: Want to discuss educational approaches or improvements? [Join the discussion](https://github.com/yourusername/STEM-EdTools/discussions)
- **Documentation**: Each tool has detailed documentation in its respective README

---

**Happy Learning! 🎓🚀** 