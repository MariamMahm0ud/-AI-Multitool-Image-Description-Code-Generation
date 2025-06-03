# app.py
import gradio as gr
from utils import ModelManager
from PIL import Image
import torch
import logging
import os
from typing import Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize model manager
try:
    logger.info("Initializing ModelManager...")
    model_manager = ModelManager()
    models_loaded_successfully = True
    logger.info("ModelManager initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize ModelManager: {str(e)}")
    models_loaded_successfully = False

def process_image_wrapper(image_pil: Optional[Image.Image], temperature: float, top_p: float, max_new_tokens: int) -> str:
    """Wrapper for image processing to handle Gradio inputs and model interaction.
    
    Args:
        image_pil: PIL Image to process
        temperature: Sampling temperature for Qwen
        top_p: Top-p sampling parameter for Qwen
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        str: Generated description or error message
    """
    if not models_loaded_successfully:
        return "Error: Models could not be loaded. Please check the server logs and ensure you have sufficient GPU memory."
    
    if image_pil is None:
        return "Error: Please upload an image."
    
    # Validate image size
    max_size = 1024  # Maximum dimension in pixels
    if max(image_pil.size) > max_size:
        return f"Error: Image is too large. Maximum dimension should be {max_size} pixels. Please resize your image."
    
    # Validate image format
    if image_pil.mode not in ['RGB', 'L']:
        return "Error: Unsupported image format. Please upload an RGB or grayscale image."
    
    try:
        # Generate initial caption using BLIP
        initial_caption = model_manager.generate_image_caption(image_pil)
        if "Error generating BLIP caption" in initial_caption:
            return f"Error: Failed to generate initial caption. Details: {initial_caption}"

        # Validate parameters
        if not (0 <= temperature <= 2.0):
            return "Error: Temperature must be between 0.0 and 2.0"
        if not (0 <= top_p <= 1.0):
            return "Error: Top P must be between 0.0 and 1.0"
        if not (50 <= max_new_tokens <= 512):
            return "Error: Max New Tokens must be between 50 and 512"

        # Enhance caption using Qwen
        enhanced_description = model_manager.enhance_caption(
            initial_caption,
            temperature=float(temperature),
            top_p=float(top_p),
            max_new_tokens=int(max_new_tokens)
        )
        
        if "Error enhancing caption" in enhanced_description:
            return f"Error: Failed to enhance caption. Details: {enhanced_description}"
            
        return f"Initial Caption (BLIP):\n{initial_caption}\n\nEnhanced Description (DeepSeek-Qwen):\n{enhanced_description}"
        
    except Exception as e:
        error_msg = f"An unexpected error occurred during image processing: {str(e)}"
        logger.error(error_msg)
        torch.cuda.empty_cache()
        return error_msg

def generate_code_from_prompt_wrapper(prompt: str, temperature: float, top_p: float, max_new_tokens: int) -> str:
    """Wrapper for code generation.
    
    Args:
        prompt: Natural language description of desired code
        temperature: Sampling temperature for Qwen
        top_p: Top-p sampling parameter for Qwen
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        str: Generated code or error message
    """
    if not models_loaded_successfully:
        return "Error: Models could not be loaded. Please check the server logs and ensure you have sufficient GPU memory."
    
    if not prompt or not prompt.strip():
        return "Error: Please enter a prompt for code generation."
    
    # Validate prompt length
    if len(prompt) > 2000:
        return "Error: Prompt is too long. Maximum length is 2000 characters."
    
    # Validate parameters
    if not (0 <= temperature <= 2.0):
        return "Error: Temperature must be between 0.0 and 2.0"
    if not (0 <= top_p <= 1.0):
        return "Error: Top P must be between 0.0 and 1.0"
    if not (50 <= max_new_tokens <= 1024):
        return "Error: Max New Tokens must be between 50 and 1024"
    
    try:
        code = model_manager.generate_code(
            prompt,
            temperature=float(temperature),
            top_p=float(top_p),
            max_new_tokens=int(max_new_tokens)
        )
        
        if "Error generating code" in code:
            return f"Error: Failed to generate code. Details: {code}"
            
        return code
        
    except Exception as e:
        error_msg = f"An unexpected error occurred during code generation: {str(e)}"
        logger.error(error_msg)
        torch.cuda.empty_cache()
        return error_msg

def create_ui() -> gr.Blocks:
    """Create and configure the Gradio interface.
    
    Returns:
        gr.Blocks: Configured Gradio interface
    """
    with gr.Blocks(theme=gr.themes.Soft(), title="AI Multitool - Image & Code") as demo:
        gr.Markdown("# 🤖 AI Multitool: Image Description & Code Generation")
        gr.Markdown("""
        This tool combines two powerful AI capabilities:
        1. **Image Description**: Uses BLIP for initial captions, enhanced by DeepSeek-Qwen for detailed descriptions
        2. **Code Generation**: Leverages DeepSeek-Qwen to generate Python code from natural language prompts
        
        Select a tab below to get started!
        """)

        if not models_loaded_successfully:
            gr.Error("Critical Error: AI Models failed to load. The application might not be functional. Please check the server logs for details.")

        with gr.Tabs() as tabs:
            # Image Description Tab
            with gr.TabItem("🖼️ Image Description"):
                gr.Markdown("""
                ### Image Description
                Upload an image to get:
                1. An initial caption from BLIP
                2. A detailed, enhanced description from DeepSeek-Qwen
                
                Adjust the advanced parameters to control the enhancement style.
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            type="pil",
                            label="Upload Image",
                            height=400
                        )
                        with gr.Accordion("Advanced Parameters", open=False):
                            img_temp = gr.Slider(
                                minimum=0.0,
                                maximum=2.0,
                                value=0.7,
                                step=0.1,
                                label="Temperature (higher = more creative)"
                            )
                            img_top_p = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.9,
                                step=0.05,
                                label="Top P (higher = more diverse)"
                            )
                            img_max_tokens = gr.Slider(
                                minimum=50,
                                maximum=512,
                                value=150,
                                step=10,
                                label="Max New Tokens"
                            )
                        img_button = gr.Button("👁️ Generate Description", variant="primary")
                    
                    with gr.Column(scale=2):
                        img_output = gr.Textbox(
                            label="Generated Description",
                            lines=15,
                            interactive=False
                        )
                
                img_button.click(
                    process_image_wrapper,
                    inputs=[image_input, img_temp, img_top_p, img_max_tokens],
                    outputs=img_output,
                    api_name="process_image"
                )
                
                gr.Examples(
                    examples=[
                        # Add example images here when available
                    ],
                    inputs=[image_input, img_temp, img_top_p, img_max_tokens],
                    outputs=img_output,
                    fn=process_image_wrapper,
                    cache_examples=True
                )
            
            # Code Generation Tab
            with gr.TabItem("💻 Python Code Generation"):
                gr.Markdown("""
                ### Python Code Generation
                Enter a natural language description of the code you want to generate.
                The AI will create clean, efficient Python code based on your prompt.
                
                Tips:
                - Be specific about function names, parameters, and expected behavior
                - Mention any specific libraries or approaches you want to use
                - For best results, use a lower temperature (0.1-0.3) for more deterministic code
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        code_input_prompt = gr.Textbox(
                            label="Enter your code generation prompt",
                            placeholder="e.g., Write a Python function to calculate the factorial of a number.",
                            lines=4
                        )
                        with gr.Accordion("Advanced Parameters", open=False):
                            code_temp = gr.Slider(
                                minimum=0.0,
                                maximum=2.0,
                                value=0.2,
                                step=0.1,
                                label="Temperature (lower = more deterministic)"
                            )
                            code_top_p = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.9,
                                step=0.05,
                                label="Top P"
                            )
                            code_max_tokens = gr.Slider(
                                minimum=50,
                                maximum=1024,
                                value=500,
                                step=50,
                                label="Max New Tokens"
                            )
                        code_button = gr.Button("💡 Generate Code", variant="primary")
                    
                    with gr.Column(scale=2):
                        code_output_display = gr.Code(
                            label="Generated Python Code",
                            language="python",
                            lines=20,
                            interactive=False
                        )
                
                code_button.click(
                    generate_code_from_prompt_wrapper,
                    inputs=[code_input_prompt, code_temp, code_top_p, code_max_tokens],
                    outputs=code_output_display,
                    api_name="generate_code"
                )
                
                gr.Examples(
                    examples=[
                        ["Create a Python function to find the sum of all even numbers in a list.", 0.1, 0.9, 500],
                        ["Write a Python script to read a CSV file named 'data.csv' and print its first 5 rows.", 0.2, 0.9, 500],
                        ["Generate a Python class 'Rectangle' with a constructor for width and height, and a method to calculate its area.", 0.3, 0.95, 500],
                    ],
                    inputs=[code_input_prompt, code_temp, code_top_p, code_max_tokens],
                    outputs=code_output_display,
                    fn=generate_code_from_prompt_wrapper,
                    cache_examples=True
                )
        
        return demo

if __name__ == "__main__":
    demo = create_ui()
    if models_loaded_successfully:
        logger.info("Launching Gradio App...")
        # Launch with settings that work in both Colab Drive and GitHub
        demo.launch(
            share=True,  # Always create a public URL
            debug=True,
            show_error=True,
            favicon_path=None,
            show_api=False,
            quiet=False,  # Changed to False to show the URL
            server_name="0.0.0.0",
            server_port=7860
        )
        
        # Print the public URL explicitly
        if hasattr(demo, 'share_url'):
            print("\nPublic URL:", demo.share_url)
    else:
        logger.error("Gradio App not launched due to model loading errors")
