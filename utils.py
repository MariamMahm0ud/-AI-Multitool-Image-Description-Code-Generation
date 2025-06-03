#utils
# utils.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import logging
import os
import re
import textwrap
import ast
from typing import Optional, Dict, Any, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_CONFIG = {
    "blip": {
        "model_name": "Salesforce/blip-image-captioning-base",
        "generation_params": {
            "max_new_tokens": 75,
            "num_beams": 5,
            "early_stopping": True,
            "length_penalty": 1.0,
            "repetition_penalty": 1.5
        }
    },
    "qwen": {
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "torch_dtype": torch.bfloat16,
        "device_map": "auto"
    }
}

# System prompts
SYSTEM_PROMPTS = {
    "image_enhancement": """You are an expert image describer. Given a brief caption of an image, expand it into a rich, vivid, and detailed description. 
Focus on:
1. Main subjects, actions, and interactions
2. Visual atmosphere, lighting, and color palette
3. Key background elements, textures, and composition
4. Maintain an objective, descriptive tone
5. Structure the description in 3-4 concise sentences
Do not include conversational filler or self-references.""",
    
    "code_generation": """You are a Python code generation expert. Your ONLY task is to generate valid, executable Python code.

CRITICAL INSTRUCTIONS - READ CAREFULLY:
1. Your ENTIRE response MUST be valid, executable Python code
2. NO OTHER TEXT IS PERMITTED - no explanations, no markdown, no XML tags, no thinking process
3. Start DIRECTLY with the first line of code (def, class, import, or first executable line)
4. Do NOT include any of these:
   - No "Here's the code:" or similar phrases
   - No <think>...</think> or other XML-like blocks
   - No markdown formatting (```python, ```)
   - No step-by-step explanations
   - No "Let me think..." or planning text
5. If the request is for a function, ONLY output that function
6. If the request is for a script, ONLY output the script
7. Include docstrings ONLY if they are standard Python docstrings within the code
8. NO conversational filler or self-references
9. NO explanations before or after the code

REMEMBER: Your response will be executed directly as Python code. Any non-code text will cause errors.""",

    "class_generation": """You are a Python OOP expert. Your task is to generate a single, complete, and syntactically correct Python class definition based STRICTLY on the user's requirements.

CRITICAL OOP INSTRUCTIONS - READ CAREFULLY:
1. Your ENTIRE response MUST be the class definition and NOTHING ELSE
2. Start DIRECTLY with 'class [Name]:' and end with the last line of the class definition
3. NO imports unless explicitly required by the class implementation and requested
4. NO example usage, test code, or any text outside the class block

METHOD TYPE RULES - FOLLOW EXACTLY:
1. Instance Methods (DEFAULT):
   - MUST use 'self' as first parameter
   - Use when method needs to access instance attributes (self.attribute)
   - Example: def method(self, ...): return self.attribute

2. @staticmethod:
   - ONLY use if method logically belongs to class but does NOT need instance/class state
   - NO 'self' or 'cls' parameter
   - Example: @staticmethod
             def utility_method(...): return result

3. @classmethod:
   - ONLY use if method needs to access class state or call other class methods
   - MUST use 'cls' as first parameter
   - Example: @classmethod
             def factory_method(cls, ...): return cls(...)

ATTRIBUTE RULES:
1. Initialize ALL instance attributes in the __init__ method, which is the standard Python constructor
2. Use proper Python naming conventions
3. Document attributes in class docstring if needed
4. Use type hints if appropriate (e.g., def __init__(self, width: float, height: float):)

LOGIC VERIFICATION:
1. Verify all calculations and operations match the user's requirements
2. Ensure method implementations are logically correct
3. Check that all required functionality is implemented
4. Verify no unnecessary features are added

REMEMBER:
- Your response will be executed directly as Python code
- Any non-code text will cause errors
- The class must be complete and self-contained
- Follow Python best practices and conventions"""
}

# Common non-code line patterns to filter out
NON_CODE_PATTERNS = [
    # Conversational prefixes (only if they start the line and aren't part of a comment)
    r"^(?!#)here's", r"^(?!#)this is", r"^(?!#)let me", r"^(?!#)i'll", r"^(?!#)the code",
    r"^(?!#)here is", r"^(?!#)below is", r"^(?!#)following", r"^(?!#)as requested",
    r"^(?!#)i will", r"^(?!#)we can", r"^(?!#)let's", r"^(?!#)first,", r"^(?!#)then,",
    # Markdown and formatting (only if they start the line)
    r"^###", r"^---", r"^===", r"^```", r"^~~~",
    # Explanatory prefixes (only if they start the line and aren't part of a comment)
    r"^(?!#)explanation:", r"^(?!#)example:", r"^(?!#)usage:", r"^(?!#)test:",
    r"^(?!#)output:", r"^(?!#)result:", r"^(?!#)note:", r"^(?!#)warning:",
    # XML-like tags (only if they start the line)
    r"^<think>", r"^</think>", r"^<code>", r"^</code>",
    # Common AI assistant phrases (only if they start the line and aren't part of a comment)
    r"^(?!#)i have", r"^(?!#)i've", r"^(?!#)i can", r"^(?!#)i will", r"^(?!#)i'll",
    r"^(?!#)here's how", r"^(?!#)here's what", r"^(?!#)let me show",
    # Numbered lists (only if they start the line and aren't part of a comment)
    r"^(?!#)\d+\.", r"^(?!#)\d+\)", r"^(?!#)[a-z]\)", r"^(?!#)[A-Z]\)",
    r"^(?!#)step \d+", r"^(?!#)step \d+:", r"^(?!#)step \d+\)"
]

# Patterns that indicate the end of a code block
CODE_END_PATTERNS = [
    # Section markers (only if they start the line)
    r"^# ---", r"^# ===", r"^# \*\*\*",
    # Numbered lists (only if they start the line and aren't part of a comment)
    r"^(?!#)\d+\.", r"^(?!#)\d+\)",
    # Lettered lists (only if they start the line and aren't part of a comment)
    r"^(?!#)[a-z]\)", r"^(?!#)[A-Z]\)",
    # Explanatory sections (only if they start the line and aren't part of a comment)
    r"^(?!#)note:", r"^(?!#)notes:", r"^(?!#)explanation:",
    # Code explanations (only if they start the line and aren't part of a comment)
    r"^(?!#)this code", r"^(?!#)the code", r"^(?!#)the function",
    r"^(?!#)the class", r"^(?!#)the script", r"^(?!#)the program",
    # Conversational starts (only if they start the line and aren't part of a comment)
    r"^(?!#)here's", r"^(?!#)this is", r"^(?!#)let me",
    r"^(?!#)i'll", r"^(?!#)i will", r"^(?!#)we can",
    # Usage instructions (only if they start the line and aren't part of a comment)
    r"^(?!#)to use", r"^(?!#)to run", r"^(?!#)to test"
]

# Python code starters that indicate the beginning of a code block
CODE_STARTERS = {
    'import': ['import ', 'from '],
    'definition': ['def ', 'class '],
    'control': ['if ', 'for ', 'while ', 'try:', 'with '],
    'async': ['async def ', 'async with ', 'async for '],
    'shebang': ['#!/usr/bin/env python', '#!/usr/bin/python'],
    'docstring': ['"""', "'''"],
    'basic': ['print(', 'print ', 'return ', 'yield ', 'break', 'continue', 'pass', 'raise ']
}

class ModelManager:
    def __init__(self):
        """Initialize the ModelManager with BLIP and Qwen models."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"ModelManager using device: {self.device}")

        try:
            # Initialize BLIP model for image captioning
            logger.info("Loading BLIP model...")
            self.blip_processor = BlipProcessor.from_pretrained(MODEL_CONFIG["blip"]["model_name"])
            self.blip_model = BlipForConditionalGeneration.from_pretrained(
                MODEL_CONFIG["blip"]["model_name"]
            ).to(self.device)
            self.blip_model.eval()
            logger.info("BLIP model loaded successfully")

            # Initialize Qwen model for text generation
            logger.info("Loading DeepSeek-Qwen model...")
            self.qwen_tokenizer = AutoTokenizer.from_pretrained(
                MODEL_CONFIG["qwen"]["model_name"],
                trust_remote_code=True
            )
            self.qwen_model = AutoModelForCausalLM.from_pretrained(
                MODEL_CONFIG["qwen"]["model_name"],
                torch_dtype=MODEL_CONFIG["qwen"]["torch_dtype"],
                device_map=MODEL_CONFIG["qwen"]["device_map"],
                trust_remote_code=True
            )
            self.qwen_model.eval()
            logger.info("DeepSeek-Qwen model loaded successfully")
            
            # Ensure pad_token_id is set
            if self.qwen_tokenizer.pad_token_id is None:
                self.qwen_tokenizer.pad_token_id = self.qwen_tokenizer.eos_token_id
                
        except Exception as e:
            logger.error(f"Failed to initialize models: {str(e)}")
            raise RuntimeError(f"Model initialization failed: {str(e)}")

    def generate_image_caption(self, image: Image.Image) -> str:
        """Generate initial caption using BLIP model.
        
        Args:
            image: PIL Image to caption
            
        Returns:
            str: Generated caption or error message
        """
        if image is None:
            return "No image provided for BLIP."
            
        try:
            inputs = self.blip_processor(images=image, return_tensors="pt").to(self.device)
            outputs = self.blip_model.generate(
                **inputs,
                **MODEL_CONFIG["blip"]["generation_params"]
            )
            caption = self.blip_processor.decode(outputs[0], skip_special_tokens=True)
            logger.info("Successfully generated BLIP caption")
            return caption.strip()
            
        except Exception as e:
            error_msg = f"Error in BLIP captioning: {str(e)}"
            logger.error(error_msg)
            return f"Error generating BLIP caption: {str(e)}"

    def enhance_caption(self, caption: str, temperature: float = 0.7, top_p: float = 0.9, max_new_tokens: int = 150) -> str:
        """Enhance the caption using DeepSeek-Qwen model.
        
        Args:
            caption: Initial BLIP caption to enhance
            temperature: Sampling temperature (0.0 to 2.0)
            top_p: Top-p sampling parameter (0.0 to 1.0)
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            str: Enhanced description or error message
        """
        if not caption:
            return "No initial caption provided for enhancement."

        messages = [
            {"role": "system", "content": SYSTEM_PROMPTS["image_enhancement"]},
            {"role": "user", "content": f"The image shows: \"{caption}\". Please provide a detailed description of this scene."}
        ]
        
        try:
            prompt_text = self.qwen_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self.qwen_tokenizer(prompt_text, return_tensors="pt").to(self.qwen_model.device)
            
            outputs = self.qwen_model.generate(
                inputs.input_ids,
                temperature=temperature if temperature > 0 else None,
                top_p=top_p if temperature > 0 else None,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.qwen_tokenizer.eos_token_id,
                do_sample=True if temperature > 0 else False
            )
            
            generated_text = self.qwen_tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            logger.info("Successfully enhanced caption")
            return generated_text.strip()
            
        except Exception as e:
            error_msg = f"Error in Qwen caption enhancement: {str(e)}"
            logger.error(error_msg)
            torch.cuda.empty_cache()
            return f"Error enhancing caption: {str(e)}"

    def generate_code(self, user_prompt: str, temperature: float = 0.2, top_p: float = 0.9, max_new_tokens: int = 300) -> str:
        """Generate Python code using DeepSeek-Qwen model.
        
        Args:
            user_prompt: Natural language description of desired code
            temperature: Sampling temperature (0.0 to 2.0)
            top_p: Top-p sampling parameter (0.0 to 1.0)
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            str: Generated code or error message
        """
        if not user_prompt:
            return "No prompt provided for code generation."

        # Detect if this is a class generation request
        is_class_request = any(keyword in user_prompt.lower() for keyword in [
            "class", "create a class", "define a class", "implement a class",
            "write a class", "make a class", "generate a class"
        ])

        # Select appropriate prompt based on request type
        system_prompt = SYSTEM_PROMPTS["class_generation"] if is_class_request else SYSTEM_PROMPTS["code_generation"]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Generate Python code for: {user_prompt}"}
        ]
        
        try:
            prompt_text = self.qwen_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self.qwen_tokenizer(prompt_text, return_tensors="pt").to(self.qwen_model.device)
            
            # Use lower temperature for class generation to ensure correctness
            gen_temperature = 0.1 if is_class_request else temperature
            
            outputs = self.qwen_model.generate(
                inputs.input_ids,
                temperature=gen_temperature if gen_temperature > 0 else None,
                top_p=top_p if gen_temperature > 0 else None,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.qwen_tokenizer.eos_token_id,
                do_sample=True if gen_temperature > 0 else False,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                early_stopping=True
            )
            
            generated_text = self.qwen_tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # Enhanced post-processing to ensure code-only output
            code = self._clean_code_output(generated_text, is_class_request)
            logger.info("Successfully generated code")
            return code
            
        except Exception as e:
            error_msg = f"Error in Qwen code generation: {str(e)}"
            logger.error(error_msg)
            torch.cuda.empty_cache()
            return f"Error generating code: {str(e)}"
            
    def _fix_indentation(self, code: str) -> str:
        """Fix indentation issues in Python code using textwrap.dedent.
        
        Args:
            code: Code string to fix
            
        Returns:
            str: Code with fixed indentation
        """
        if not code.strip():
            logger.debug("Empty code string received for indentation fixing")
            return code
            
        logger.debug("Applying textwrap.dedent() for indentation fixing")
        try:
            return textwrap.dedent(code)
        except Exception as e:
            logger.warning(f"Failed to dedent code: {str(e)}")
            return code

    def _test_code_extraction(self, text: str) -> None:
        """Test method to debug code extraction process.
        
        Args:
            text: Raw text to test extraction on
        """
        logger.info("=== Starting Code Extraction Test ===")
        logger.info(f"Input text:\n{text}")
        
        # Test markdown removal
        if text.strip().startswith("```python"):
            text = text.split("```python\n", 1)[-1]
            logger.info("Found and removed ```python markdown block")
        if text.strip().startswith("```"):
            text = text.split("```\n", 1)[-1]
            logger.info("Found and removed ``` markdown block")
        if text.strip().endswith("```"):
            text = text.rsplit("\n```", 1)[0]
            logger.info("Found and removed trailing ``` markdown block")
            
        # Test XML block removal
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        if '<think>' in text:
            logger.info("Found and removed <think> XML block")
            
        # Split and analyze lines
        lines = text.split("\n")
        logger.info(f"Processing {len(lines)} lines")
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                logger.info(f"Line {i+1}: [empty line]")
                continue
                
            # Test code starter detection
            for starter in [kw for keywords in CODE_STARTERS.values() for kw in keywords]:
                if line.startswith(starter):
                    logger.info(f"Line {i+1}: Found code starter '{starter}'")
                    break
            else:
                # Test non-code pattern matching
                for pattern in NON_CODE_PATTERNS:
                    if re.match(pattern, line.lower()):
                        logger.info(f"Line {i+1}: Matched non-code pattern '{pattern}'")
                        break
                else:
                    logger.info(f"Line {i+1}: No pattern match - potential code line")
                    
        logger.info("=== End Code Extraction Test ===")

    def _clean_code_output(self, text: str, is_class: bool = False) -> str:
        """Clean generated code output by finding the largest syntactically valid Python code block.
        
        This function uses an iterative approach to find the largest contiguous block of text
        that is valid Python code according to ast.parse(). It starts with minimal cleaning
        of markdown and XML wrappers, then systematically tries different line ranges to find
        the longest valid code block.
        
        Args:
            text: Raw generated text
            is_class: Whether the output should be a class definition (not used in new approach)
            
        Returns:
            str: Cleaned code or error message if no valid code found
        """
        if not text or not text.strip():
            logger.error("Empty input text")
            return "Error: No valid Python code could be extracted from the model's output"

        # Step 1: Initial wrapper removal (minimal cleaning)
        logger.debug("Starting initial wrapper removal")
        
        # Remove markdown code blocks
        if text.strip().startswith("```python"):
            text = text.split("```python\n", 1)[-1]
            logger.debug("Removed ```python markdown block")
        if text.strip().startswith("```"):
            text = text.split("```\n", 1)[-1]
            logger.debug("Removed ``` markdown block")
        if text.strip().endswith("```"):
            text = text.rsplit("\n```", 1)[0]
            logger.debug("Removed trailing ``` markdown block")
            
        # Remove XML-like think blocks
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        if '<think>' in text:
            logger.debug("Removed <think> XML block")
        
        # Step 2: Split into lines and initialize variables
        lines = text.split("\n")
        logger.debug(f"Processing {len(lines)} lines after initial cleaning")
        
        best_valid_code = ""
        max_valid_lines = 0
        
        # Step 3: Iterative block finding and parsing
        for start_index in range(len(lines)):
            # Skip empty lines at the start
            if not lines[start_index].strip():
                continue
            
            # Try to find the longest valid block starting from this line
            current_block_lines = []
            current_block = ""
            
            for end_index in range(start_index, len(lines)):
                # Add the current line to our block
                current_block_lines.append(lines[end_index])
                current_block = "\n".join(current_block_lines)
                
                # Skip if the block is empty after stripping
                if not current_block.strip():
                    continue
                
                # Try to parse the current block
                try:
                    # Dedent the code before parsing
                    dedented_code = textwrap.dedent(current_block)
                    
                    # Try to parse the code
                    ast.parse(dedented_code)
                    
                    # If we get here, the code is valid
                    if len(current_block_lines) > max_valid_lines:
                        max_valid_lines = len(current_block_lines)
                        best_valid_code = dedented_code.strip()
                        logger.debug(f"Found new best valid code block: {len(current_block_lines)} lines")
                        
                except SyntaxError:
                    # This block ending here wasn't valid, but we might find a valid block
                    # starting from the same start_index with a different end_index
                    continue
                except Exception as e:
                    # Log unexpected errors but continue searching
                    logger.warning(f"Unexpected error while parsing block: {str(e)}")
                    continue
        
        # Step 4: Return the result
        if best_valid_code:
            logger.info(f"Successfully extracted valid Python code block of {max_valid_lines} lines")
            return best_valid_code
        else:
            logger.error("No valid Python code block could be found")
            return "Error: No valid Python code could be extracted from the model's output"