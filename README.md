# AI Multitool: Image Description & Code Generation

A Gradio-based application that combines image description and code generation capabilities using the DeepSeek-R1-Distill-Qwen-1.5B model.

## Features

### 1. Image Description
- Upload images to get detailed descriptions
- Uses BLIP for initial captions
- Enhanced by DeepSeek-Qwen for rich, detailed descriptions
- Adjustable parameters for description style and length

### 2. Code Generation
- Generate Python code from natural language prompts
- Supports functions, classes, and complete scripts
- Intelligent code extraction and cleaning
- Adjustable parameters for code generation

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- 12GB+ VRAM for optimal performance

### Installation

1. Clone the repository:
```bash
git clone https://github.com/MariamMahm0ud/-AI-Multitool-Image-Description-Code-Generation
cd -AI-Multitool-Image-Description-Code-Generation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

## Usage Guide

### Image Description
1. Navigate to the "Image Description" tab
2. Upload an image using the file uploader
3. Adjust parameters if needed:
   - Temperature (0.0-2.0): Higher values for more creative descriptions
   - Top P (0.0-1.0): Higher values for more diverse outputs
   - Max New Tokens (50-512): Control description length
4. Click "Generate Description" to process the image

### Code Generation
1. Navigate to the "Code Generation" tab
2. Enter your code generation prompt
3. Adjust parameters if needed:
   - Temperature (0.0-2.0): Lower values (0.1-0.3) for more deterministic code
   - Top P (0.0-1.0): Higher values for more diverse outputs
   - Max New Tokens (50-1024): Control code length
4. Click "Generate Code" to create the code

## Parameter Tuning Guide

### Image Description Parameters
- **Temperature**: 
  - 0.7 (default): Balanced creativity and coherence
  - < 0.5: More focused, literal descriptions
  - > 1.0: More creative, varied descriptions

- **Top P**:
  - 0.9 (default): Good balance of diversity and quality
  - < 0.7: More focused outputs
  - > 0.95: More diverse outputs

- **Max New Tokens**:
  - 150 (default): Standard description length
  - < 100: Brief descriptions
  - > 200: Detailed descriptions

### Code Generation Parameters
- **Temperature**:
  - 0.2 (default): Good for most code generation
  - 0.1: For precise, deterministic code
  - 0.3: For more creative solutions

- **Top P**:
  - 0.9 (default): Balanced diversity
  - 0.95: For more varied solutions
  - 0.8: For more focused code

- **Max New Tokens**:
  - 500 (default): Suitable for most code snippets
  - < 300: For simple functions
  - > 700: For complex classes or scripts

## Example Prompts

### Image Description
- "Describe this image in detail"
- "What's happening in this scene?"
- "Describe the main elements and their relationships"

### Code Generation
1. Simple Function:
```
Create a Python function to find the sum of all even numbers in a list.
```

2. File Processing:
```
Write a Python script to read a CSV file named 'data.csv' and print its first 5 rows.
```

3. Class Implementation:
```
Generate a Python class 'Rectangle' with a constructor for width and height, and a method to calculate its area.
```

## Troubleshooting

### Common Issues
1. **Model Loading Errors**
   - Ensure sufficient GPU memory
   - Check internet connection for model download
   - Verify CUDA installation

2. **Generation Errors**
   - Try reducing max_new_tokens
   - Adjust temperature and top_p
   - Check input format and size

3. **Performance Issues**
   - Reduce image size before upload
   - Close other GPU applications
   - Use smaller batch sizes
# AI Multitool: Image Description & Code Generation

A Gradio-based application that combines image description and code generation capabilities using the DeepSeek-R1-Distill-Qwen-1.5B model.

## Features

### 1. Image Description
- Upload images to get detailed descriptions
- Uses BLIP for initial captions
- Enhanced by DeepSeek-Qwen for rich, detailed descriptions
- Adjustable parameters for description style and length

### 2. Code Generation
- Generate Python code from natural language prompts
- Supports functions, classes, and complete scripts
- Intelligent code extraction and cleaning
- Adjustable parameters for code generation

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- 12GB+ VRAM for optimal performance

### Installation

1. Clone the repository:
```bash
git clone https://github.com/MariamMahm0ud/-AI-Multitool-Image-Description-Code-Generation
cd -AI-Multitool-Image-Description-Code-Generation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

## Usage Guide

### Image Description
1. Navigate to the "Image Description" tab
2. Upload an image using the file uploader
3. Adjust parameters if needed:
   - Temperature (0.0-2.0): Higher values for more creative descriptions
   - Top P (0.0-1.0): Higher values for more diverse outputs
   - Max New Tokens (50-512): Control description length
4. Click "Generate Description" to process the image

### Code Generation
1. Navigate to the "Code Generation" tab
2. Enter your code generation prompt
3. Adjust parameters if needed:
   - Temperature (0.0-2.0): Lower values (0.1-0.3) for more deterministic code
   - Top P (0.0-1.0): Higher values for more diverse outputs
   - Max New Tokens (50-1024): Control code length
4. Click "Generate Code" to create the code

## Parameter Tuning Guide

### Image Description Parameters
- **Temperature**: 
  - 0.7 (default): Balanced creativity and coherence
  - < 0.5: More focused, literal descriptions
  - > 1.0: More creative, varied descriptions

- **Top P**:
  - 0.9 (default): Good balance of diversity and quality
  - < 0.7: More focused outputs
  - > 0.95: More diverse outputs

- **Max New Tokens**:
  - 150 (default): Standard description length
  - < 100: Brief descriptions
  - > 200: Detailed descriptions

### Code Generation Parameters
- **Temperature**:
  - 0.2 (default): Good for most code generation
  - 0.1: For precise, deterministic code
  - 0.3: For more creative solutions

- **Top P**:
  - 0.9 (default): Balanced diversity
  - 0.95: For more varied solutions
  - 0.8: For more focused code

- **Max New Tokens**:
  - 500 (default): Suitable for most code snippets
  - < 300: For simple functions
  - > 700: For complex classes or scripts

## Example Prompts

### Image Description
- "Describe this image in detail"
- "What's happening in this scene?"
- "Describe the main elements and their relationships"

### Code Generation
1. Simple Function:
```
Create a Python function to find the sum of all even numbers in a list.
```

2. File Processing:
```
Write a Python script to read a CSV file named 'data.csv' and print its first 5 rows.
```

3. Class Implementation:
```
Generate a Python class 'Rectangle' with a constructor for width and height, and a method to calculate its area.
```

## Troubleshooting

### Common Issues
1. **Model Loading Errors**
   - Ensure sufficient GPU memory
   - Check internet connection for model download
   - Verify CUDA installation

2. **Generation Errors**
   - Try reducing max_new_tokens
   - Adjust temperature and top_p
   - Check input format and size

3. **Performance Issues**
   - Reduce image size before upload
   - Close other GPU applications
   - Use smaller batch sizes

# Conclusion

In this project, each model was carefully selected to fulfill a distinct role and ensure optimal performance across tasks:

BLIP (Bootstrapped Language Image Pretraining) was used for initial image captioning due to its strong vision-language alignment and fast inference. It provides a concise and relevant base caption from raw images, which serves as input to further enhance with language models.

DeepSeek-R1-Distill-Qwen-1.5B was chosen for its balance between speed and performance. It refines image captions into rich, detailed descriptions and also powers natural language to Python code generation with impressive contextual understanding. Its distilled nature makes it efficient for local deployment with limited GPU resources.

CLIP (Contrastive Language–Image Pretraining) is optionally included to enable future extensions like zero-shot image classification or semantic filtering based on visual-textual similarity.

This combination allows the AI Multitool to perform high-quality multimodal tasks—image understanding 
