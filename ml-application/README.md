# ML Context Capture & Analysis Tool

A machine learning demonstration application that captures screen content through a transparent overlay window and uses it as context for OpenAI's GPT-4 Vision model.

## Features

### 🎯 **Interactive Screen Capture**
- **Transparent overlay**: Draw a selection rectangle over any part of your screen
- **Real-time preview**: See your selection area highlighted in real-time
- **Keyboard shortcuts**: Press Enter to capture, Escape to cancel
- **High-quality capture**: Uses MSS library for fast, accurate screen capture

### 🤖 **AI-Powered Analysis**
- **GPT-4 Vision integration**: Sends captured images to OpenAI's vision model
- **Contextual prompts**: Ask questions about what the AI sees in the image
- **Real-time responses**: Get detailed analysis and insights
- **Example prompts**: Built-in suggestions for common use cases

### 🎨 **User-Friendly Interface**
- **Tabbed layout**: Clean separation between controls and results
- **Visual feedback**: Status indicators and progress updates
- **Responsive design**: Adapts to different screen sizes
- **Professional appearance**: Modern, intuitive interface

## Installation

### Prerequisites
- Python 3.8+
- OpenAI API key

### Setup
1. **Clone or download the project**
2. **Install dependencies**:
   ```bash
   cd STEM-EdTools
   pip install -r requirements.txt
   ```

3. **Set up your OpenAI API key**:
   Create a `.env` file in the project directory:
   ```
   OPENAI_KEY=your_openai_api_key_here
   ```

4. **Run the application**:
   ```bash
   cd ml-application
   python main.py
   ```

## Usage

### Basic Workflow
1. **Start Capture**: Click "🎯 Start Screen Capture"
2. **Select Area**: Click and drag to draw a rectangle over the desired area
3. **Capture**: Press Enter to capture the selected area
4. **Enter Prompt**: Type your question or prompt about the captured content
5. **Get Analysis**: Click "🚀 Send to AI" to receive AI analysis

### Example Use Cases

#### **Document Analysis**
- Capture a document or report
- Ask: "Summarize the main points of this document"
- Get: Structured summary with key insights

#### **UI/UX Testing**
- Capture a website or application interface
- Ask: "What improvements could be made to this interface?"
- Get: Detailed UX recommendations

#### **Content Understanding**
- Capture any visual content
- Ask: "What is happening in this image?"
- Get: Detailed description and analysis

#### **Problem Solving**
- Capture error messages or technical issues
- Ask: "What does this error mean and how can I fix it?"
- Get: Explanation and solution suggestions

### Example Prompts

The application includes built-in example prompts:
- "What do you see in this image?"
- "Describe the content in detail"
- "What is the main topic or subject?"
- "Analyze the layout and structure"
- "What actions could I take based on this?"

## Technical Details

### **Screen Capture Technology**
- **MSS Library**: Fast, efficient screen capture
- **PIL/Pillow**: Image processing and manipulation
- **Tkinter**: GUI framework for overlay window

### **AI Integration**
- **OpenAI GPT-4 Vision**: State-of-the-art vision model
- **Base64 encoding**: Secure image transmission
- **Async processing**: Non-blocking AI requests

### **Image Processing**
- **Automatic resizing**: Images scaled to fit display
- **Aspect ratio preservation**: Maintains image proportions
- **High-quality capture**: Lossless PNG format

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│ ML Context Capture & Analysis Tool                      │
├─────────────────────────────────────────────────────────┤
│ ┌─ Controls Panel ─────────────────────────────────────┐ │
│ │ 🎯 Start Screen Capture                              │ │
│ │ 📝 Prompt Input                                      │ │
│ │ 🚀 Send to AI                                        │ │
│ │ 📊 Status Display                                    │ │
│ └─────────────────────────────────────────────────────┘ │
│ ┌─ Display Panel ──────────────────────────────────────┐ │
│ │ 📸 Captured Image                                    │ │
│ │ 💬 AI Response                                       │ │
│ └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## Dependencies

All dependencies are managed in the root `requirements.txt` file:
- **opencv-python**: Computer vision operations
- **numpy**: Numerical computing
- **Pillow**: Image processing
- **openai**: OpenAI API client
- **python-dotenv**: Environment variable management
- **pyautogui**: Screen capture utilities
- **mss**: Fast screen capture
- **tkinter**: GUI framework

## Educational Value

This application demonstrates several important ML/AI concepts:

### **Computer Vision**
- Screen capture and image processing
- Visual context understanding
- Image-to-text analysis

### **AI Integration**
- API communication with OpenAI
- Prompt engineering
- Context-aware responses

### **User Experience**
- Intuitive interface design
- Real-time feedback
- Error handling and recovery

### **Practical Applications**
- Document analysis automation
- UI/UX testing assistance
- Content understanding tools
- Problem diagnosis and solving

## Troubleshooting

### **Common Issues**

**"OPENAI_KEY not found"**
- Ensure your `.env` file exists and contains the correct API key
- Check that the key is properly formatted without quotes

**"Failed to capture screen"**
- Ensure you have proper screen capture permissions
- Try running as administrator if on Windows
- Check that no other applications are blocking screen capture

**"Error communicating with AI"**
- Verify your OpenAI API key is valid and has credits
- Check your internet connection
- Ensure you're using a supported model

### **Performance Tips**
- Use smaller capture areas for faster processing
- Close unnecessary applications to reduce memory usage
- Ensure stable internet connection for AI requests

## Future Enhancements

Potential improvements for future versions:
- **Multiple capture areas**: Capture and analyze multiple regions
- **Video capture**: Support for moving content analysis
- **Custom models**: Integration with other AI models
- **Batch processing**: Analyze multiple images at once
- **Export functionality**: Save captures and responses
- **Advanced prompts**: Template-based prompt generation

## License

This project is part of the STEM-EdTools collection and is licensed under the MIT License.

## Contributing

Contributions are welcome! Areas for improvement:
- Additional AI model integrations
- Enhanced UI/UX features
- Performance optimizations
- Educational content and examples 