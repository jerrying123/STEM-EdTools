# Sampling Biases Learning Tool

An interactive GUI application that teaches about sampling biases and data representativeness in machine learning using the MNIST handwritten digit dataset.

## Purpose

This educational tool demonstrates how different types of sampling biases and dataset composition choices affect machine learning model performance. Students can:

- Select which digit classes (0-9) to include in their dataset
- Control how many samples from each class to use via sliders
- Train a neural network on their custom dataset
- Test the model and see how sampling biases affect performance
- Understand various types of bias in data collection and preparation

## Features

- **Interactive Class Selection**: Click on digit visualizations to select which classes to include
- **Dynamic Dataset Control**: Use sliders to control the number of samples per class (10-1000 samples)
- **Real-time Training**: Watch model training progress with live updates
- **Performance Analysis**: Detailed test results showing per-class accuracy
- **Educational Insights**: Clear demonstration of how sampling biases affect model performance
- **Bias Type Identification**: Learn about selection bias, measurement bias, and sampling bias

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd STEM-EdTools
```

2. Install the required dependencies:
```bash
cd sampling-biases
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python main.py
```

2. Wait for the MNIST dataset to load (first time only - it will be downloaded automatically)

3. **Select digit classes**: Click on the digit images on the left panel to select which classes you want to include in your dataset. Selected classes will be highlighted in blue.

4. **Adjust dataset sizes**: For each selected class, use the sliders in the right panel to control how many samples to include (10-1000 per class).

5. **Create dataset**: Click "Create Dataset" to generate your custom train/validation/test splits.

6. **Train the model**: Click "Start Training" to begin training the neural network. You can monitor progress in the text area.

7. **Test performance**: After training completes, click "Test Model" to evaluate performance and see per-class results.

## Educational Experiments

Try these experiments to learn about sampling biases:

### Experiment 1: Selection Bias - Balanced vs Imbalanced
- **Balanced**: Select all digits (0-9) with equal sample sizes (e.g., 500 each)
- **Imbalanced**: Select all digits but vary sample sizes dramatically (e.g., digit 0: 50, digit 1: 1000)
- Compare the per-class accuracies to see how selection bias affects performance

### Experiment 2: Sampling Bias - Missing Classes
- Train on only digits 0-4, then test (the test set will still contain all digits)
- Observe how the model performs on unseen digit classes
- This demonstrates sampling bias where certain classes are systematically excluded

### Experiment 3: Sample Size Bias - Size Effects
- Select the same classes but vary total dataset size
- Small dataset: 50 samples per class
- Large dataset: 800 samples per class
- Compare overall performance and training stability

### Experiment 4: Measurement Bias - Feature Selection
- Train models with different feature subsets
- Compare performance to understand how feature selection can introduce bias

## Types of Sampling Bias Demonstrated

### Selection Bias
- **Definition**: When certain classes or samples are systematically excluded from the dataset
- **Example**: Only including digits 0-4 in training data
- **Impact**: Poor performance on excluded classes

### Sampling Bias
- **Definition**: When the sampling method favors certain types of data
- **Example**: Having 1000 samples of digit '1' but only 50 of digit '7'
- **Impact**: Model becomes biased toward overrepresented classes

### Measurement Bias
- **Definition**: When the way data is collected or processed introduces systematic errors
- **Example**: Using different preprocessing for different digit classes
- **Impact**: Inconsistent feature representations

## Technical Details

- **Model**: Convolutional Neural Network with 2 conv layers and 2 fully connected layers
- **Framework**: PyTorch for deep learning, Tkinter for GUI
- **Dataset**: MNIST handwritten digits (automatically downloaded)
- **Training**: 5 epochs with Adam optimizer
- **Evaluation**: Accuracy and loss metrics, both overall and per-class

## File Structure

- `main.py`: Main GUI application
- `model.py`: Neural network model definition
- `data_handler.py`: MNIST data loading and custom dataset creation
- `trainer.py`: Model training and testing logic
- `requirements.txt`: Python dependencies

## Learning Outcomes

After using this tool, students will understand:

1. **Sampling biases**: How different types of bias affect model performance
2. **Selection bias**: Why excluding certain classes leads to poor generalization
3. **Class imbalance**: Why balanced datasets generally lead to better performance
4. **Sample size effects**: How more training data typically improves model performance
5. **Generalization**: Why models perform poorly on unseen classes or underrepresented classes
6. **Bias detection**: How to identify and measure different types of sampling bias

## Requirements

- Python 3.7+
- PyTorch 2.0+
- Tkinter (usually included with Python)
- PIL/Pillow for image processing
- NumPy and Matplotlib

## Troubleshooting

**Issue**: GUI doesn't appear or looks broken
**Solution**: Ensure you have tkinter installed. On some Linux distributions: `sudo apt-get install python3-tk`

**Issue**: MNIST download fails
**Solution**: Check your internet connection. The dataset (~50MB) will be automatically downloaded to a `./data` folder.

**Issue**: Training is slow
**Solution**: If you have a CUDA-capable GPU, PyTorch will automatically use it. Otherwise, training runs on CPU which is slower but still functional for this educational tool.

## Contributing

This is an educational tool. Feel free to extend it with additional features like:
- Different model architectures
- Additional datasets (CIFAR-10, Fashion-MNIST)
- More detailed visualization of training progress
- Export functionality for results
- Additional bias types and demonstrations

## License

Open source - feel free to use and modify for educational purposes. 