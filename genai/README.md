# GenAI Image Generation Tool

An interactive GUI application for generating images using state-of-the-art AI models from Hugging Face. This tool demonstrates the power of generative AI and provides an educational platform for understanding text-to-image generation.

## Features

### 🎨 **Multiple AI Models**
- **Stable Diffusion**: High-quality, photorealistic image generation
- **DALL-E Mini**: Fast, creative image generation
- **Custom Model Support**: Framework for adding new models

### 🖼️ **Advanced Generation Controls**
- **Custom Prompts**: Enter detailed descriptions for image generation
- **Multiple Images**: Generate 1-4 images per prompt
- **Size Options**: 256x256 to 1024x1024 resolution
- **Quality Settings**: Adjustable steps and guidance scale
- **Real-time Parameters**: Fine-tune generation on the fly

### 🎯 **User-Friendly Interface**
- **Interactive Gallery**: Visual display of generated images
- **Example Prompts**: Built-in suggestions for common use cases
- **Generation Log**: Detailed logging of the generation process
- **Progress Tracking**: Real-time status updates
- **Batch Operations**: Save all images or clear gallery

### 📊 **Educational Features**
- **Model Comparison**: Compare different AI models side-by-side
- **Parameter Exploration**: Understand how settings affect output
- **Visual Learning**: See AI capabilities in real-time
- **Statistics Tracking**: Monitor generation performance

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for best performance)
- 8GB+ RAM (for model loading)

### Setup
1. **Install dependencies**:
   ```bash
   cd STEM-EdTools
   pip install -r requirements.txt
   ```

2. **Run the application**:
   ```bash
   cd genai
   python main.py
   ```

## Usage

### Basic Workflow
1. **Select Model**: Choose between Stable Diffusion or DALL-E Mini
2. **Enter Prompt**: Describe the image you want to generate
3. **Adjust Parameters**: Set image size, quality, and other options
4. **Generate**: Click "🎨 Generate Images" to start
5. **View Results**: Images appear in the gallery
6. **Save Images**: Use "💾 Save All" to download your creations

### Example Prompts

The application includes built-in example prompts:
- "A beautiful sunset over mountains"
- "A futuristic city with flying cars"
- "A cute cat wearing a space helmet"
- "Abstract art with vibrant colors"
- "A steampunk robot in a Victorian setting"

### Advanced Usage

#### **Parameter Tuning**
- **Steps**: Higher values (50-100) = better quality, slower generation
- **Guidance Scale**: Higher values (7-15) = more adherence to prompt
- **Image Size**: Larger sizes = more detail, slower generation

#### **Model Selection**
- **Stable Diffusion**: Best for photorealistic images, requires more GPU memory
- **DALL-E Mini**: Faster generation, good for creative/artistic images

## Technical Details

### **Supported Models**
- **Stable Diffusion v1.5**: Industry-standard text-to-image model
- **DALL-E Mini**: Lightweight alternative for quick generation
- **Extensible Architecture**: Easy to add new models

### **Generation Pipeline**
```
Text Prompt → Model Processing → Image Generation → Display & Save
```

### **Performance Optimization**
- **GPU Acceleration**: Automatic CUDA detection and usage
- **Memory Management**: Efficient model loading and cleanup
- **Batch Processing**: Generate multiple images efficiently

## Educational Value

This tool demonstrates several important AI/ML concepts:

### **Generative AI**
- Text-to-image generation using transformer models
- Understanding of diffusion models and their applications
- Prompt engineering and its impact on output quality

### **Model Comparison**
- Different approaches to image generation
- Trade-offs between speed and quality
- Parameter tuning and its effects

### **Practical Applications**
- Creative content generation
- AI-assisted design workflows
- Understanding AI capabilities and limitations

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│ GenAI Image Generation Tool                              │
├─────────────────────────────────────────────────────────┤
│ ┌─ Controls Panel ─────────────────────────────────────┐ │
│ │ 🎨 Model Selection                                   │ │
│ │ 📝 Prompt Input                                      │ │
│ │ ⚙️ Parameters                                        │ │
│ │ 🚀 Generate Button                                   │ │
│ └─────────────────────────────────────────────────────┘ │
│ ┌─ Display Panel ──────────────────────────────────────┐ │
│ │ 🖼️ Image Gallery                                     │ │
│ │ 📊 Generation Log                                    │ │
│ └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## Dependencies

All dependencies are managed in the root `requirements.txt` file:
- **transformers**: Hugging Face model library
- **diffusers**: Diffusion model implementations
- **torch**: PyTorch deep learning framework
- **Pillow**: Image processing
- **matplotlib**: Visualization utilities

## Troubleshooting

### **Common Issues**

**"CUDA out of memory"**
- Reduce image size or number of images
- Use CPU mode (slower but works)
- Close other GPU-intensive applications

**"Model loading failed"**
- Check internet connection (models download on first use)
- Ensure sufficient disk space (models are large)
- Try restarting the application

**"Generation is slow"**
- Use smaller image sizes
- Reduce number of steps
- Ensure GPU is being used (check logs)

### **Performance Tips**
- Use GPU for best performance
- Start with smaller images for testing
- Close unnecessary applications to free memory
- Use fewer steps for faster generation

## Future Enhancements

Potential improvements for future versions:
- **More Models**: Add support for additional AI models
- **Image Editing**: Inpainting and outpainting capabilities
- **Style Transfer**: Apply artistic styles to generated images
- **Batch Processing**: Generate images from text files
- **API Integration**: Connect to cloud-based generation services
- **Advanced Controls**: More fine-grained parameter adjustment

## Educational Use Cases

### **For Students**
- Learn about generative AI and its capabilities
- Understand the relationship between prompts and outputs
- Explore different AI models and their strengths
- Practice prompt engineering techniques

### **For Educators**
- Demonstrate AI capabilities in real-time
- Show the creative potential of AI
- Teach about model parameters and their effects
- Compare different approaches to image generation

### **For Researchers**
- Test and compare different models
- Experiment with prompt engineering
- Understand model limitations and biases
- Prototype AI-powered applications

## License

This project is part of the STEM-EdTools collection and is licensed under the MIT License.

## Contributing

Contributions are welcome! Areas for improvement:
- Additional AI model integrations
- Enhanced UI/UX features
- Performance optimizations
- Educational content and tutorials
- New generation techniques

## Acknowledgments

- **Hugging Face** for providing the model library and infrastructure
- **Stability AI** for the Stable Diffusion model
- **OpenAI** for DALL-E research and inspiration
- **The open-source community** for making AI accessible
