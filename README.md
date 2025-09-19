# 🚀 AI Multitool: Image Description & Code Generation

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![Gradio](https://img.shields.io/badge/gradio-4.19.2%2B-orange.svg)](https://gradio.app)
[![PyTorch](https://img.shields.io/badge/pytorch-2.2.0%2B-red.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/MariamMahm0ud/-AI-Multitool-Image-Description-Code-Generation.svg)](https://github.com/MariamMahm0ud/-AI-Multitool-Image-Description-Code-Generation/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/MariamMahm0ud/-AI-Multitool-Image-Description-Code-Generation.svg)](https://github.com/MariamMahm0ud/-AI-Multitool-Image-Description-Code-Generation/network)

> A powerful Gradio-based application that combines image description and code generation capabilities using state-of-the-art AI models including BLIP and DeepSeek-R1-Distill-Qwen-1.5B.

## 🌟 Features

### 🖼️ Image Description
- Upload images to get detailed descriptions
- Uses BLIP for initial captions
- Enhanced by DeepSeek-Qwen for rich, detailed descriptions
- Adjustable parameters for description style and length

![Image Description Demo](https://github.com/MariamMahm0ud/-AI-Multitool-Image-Description-Code-Generation/blob/master/Screenshot%202025-06-04%20014226.png?raw=true)

### 💻 Code Generation
- Generate Python code from natural language prompts
- Supports functions, classes, and complete scripts
- Intelligent code extraction and cleaning
- Adjustable parameters for code generation

![Code Generation Demo](https://github.com/MariamMahm0ud/-AI-Multitool-Image-Description-Code-Generation/blob/master/Screenshot%202025-06-04%20014124.png?raw=true)

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- 12GB+ VRAM for optimal performance

### Installation

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fu_mFWnCUq2Gz8FBKe0BGrpWW_FqFQ8L#scrollTo=EPIv5LgA1bGm)

1. **Clone the repository:**
```bash
git clone https://github.com/MariamMahm0ud/-AI-Multitool-Image-Description-Code-Generation
cd -AI-Multitool-Image-Description-Code-Generation
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the application:**
```bash
python app.py
```

## 📖 Usage Guide

### Image Description
1. Navigate to the "🖼️ Image Description" tab
2. Upload an image using the file uploader
3. Adjust parameters if needed:
   - **Temperature** (0.0-2.0): Higher values for more creative descriptions
   - **Top P** (0.0-1.0): Higher values for more diverse outputs
   - **Max New Tokens** (50-512): Control description length
4. Click "👁️ Generate Description" to process the image

### Code Generation
1. Navigate to the "💻 Code Generation" tab
2. Enter your code generation prompt
3. Adjust parameters if needed:
   - **Temperature** (0.0-2.0): Lower values (0.1-0.3) for more deterministic code
   - **Top P** (0.0-1.0): Higher values for more diverse outputs
   - **Max New Tokens** (50-1024): Control code length
4. Click "💡 Generate Code" to create the code

## ⚙️ Parameter Tuning Guide

### Image Description Parameters
- **Temperature**: 
  - `0.7` (default): Balanced creativity and coherence
  - `< 0.5`: More focused, literal descriptions
  - `> 1.0`: More creative, varied descriptions

- **Top P**:
  - `0.9` (default): Good balance of diversity and quality
  - `< 0.7`: More focused outputs
  - `> 0.95`: More diverse outputs

- **Max New Tokens**:
  - `150` (default): Standard description length
  - `< 100`: Brief descriptions
  - `> 200`: Detailed descriptions

### Code Generation Parameters
- **Temperature**:
  - `0.2` (default): Good for most code generation
  - `0.1`: For precise, deterministic code
  - `0.3`: For more creative solutions

- **Top P**:
  - `0.9` (default): Balanced diversity
  - `0.95`: For more varied solutions
  - `0.8`: For more focused code

- **Max New Tokens**:
  - `500` (default): Suitable for most code snippets
  - `< 300`: For simple functions
  - `> 700`: For complex classes or scripts

## 💡 Example Prompts

### Image Description
- "Describe this image in detail"
- "What's happening in this scene?"
- "Describe the main elements and their relationships"

### Code Generation

#### 1. Simple Function:
```
Create a Python function to find the sum of all even numbers in a list.
```

#### 2. File Processing:
```
Write a Python script to read a CSV file named 'data.csv' and print its first 5 rows.
```

#### 3. Class Implementation:
```
Generate a Python class 'Rectangle' with a constructor for width and height, and a method to calculate its area.
```

## 🏗️ Architecture

The AI Multitool uses a sophisticated architecture combining multiple state-of-the-art models:

- **BLIP** (Bootstrapped Language Image Pretraining): Provides fast and accurate initial image captions
- **DeepSeek-R1-Distill-Qwen-1.5B**: Enhances captions and generates Python code with excellent contextual understanding
- **Gradio Interface**: Modern web interface for seamless user interaction

## 🔧 Troubleshooting

### Common Issues

#### 1. Model Loading Errors
- Ensure sufficient GPU memory
- Check internet connection for model download
- Verify CUDA installation

#### 2. Generation Errors
- Try reducing `max_new_tokens`
- Adjust `temperature` and `top_p`
- Check input format and size

#### 3. Performance Issues
- Reduce image size before upload
- Close other GPU applications
- Use smaller batch sizes

## 📋 Technical Requirements

- **Python**: 3.8+
- **GPU**: CUDA-capable (recommended)
- **VRAM**: 12GB+ for optimal performance
- **Dependencies**: See `requirements.txt`

## 📚 Documentation

For detailed technical documentation, architecture details, and API reference, see [DOCUMENTATION.md](DOCUMENTATION.md).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 👤 Author

**Mariam Mahmoud**
- GitHub: [@MariamMahm0ud](https://github.com/MariamMahm0ud)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 Acknowledgments

- **BLIP Model**: Salesforce Research for the excellent image captioning model
- **DeepSeek**: For the powerful language model
- **Gradio**: For the amazing web interface framework
- **Hugging Face**: For the transformers library and model hosting

## 📊 Project Stats

![GitHub repo size](https://img.shields.io/github/repo-size/MariamMahm0ud/-AI-Multitool-Image-Description-Code-Generation)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/MariamMahm0ud/-AI-Multitool-Image-Description-Code-Generation)
![GitHub last commit](https://img.shields.io/github/last-commit/MariamMahm0ud/-AI-Multitool-Image-Description-Code-Generation)

---

⭐ **Star this repository if you found it helpful!**

*Made with ❤️ by [Mariam Mahmoud](https://github.com/MariamMahm0ud)*