from utils import ModelManager
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_code_extraction():
    """Test the code extraction logic with various examples."""
    # Initialize ModelManager
    model_manager = ModelManager()
    
    # Test cases
    test_cases = [
        # Test Case 1: Simple Python code
        {
            "name": "Simple Python code",
            "input": """print('Hello, World!')"""
        },
        
        # Test Case 2: Code with markdown wrappers
        {
            "name": "Code with markdown",
            "input": """Here's the code:
```python
def greet(name):
    return f"Hello, {name}!"
```
You can use it like this."""
        },
        
        # Test Case 3: Code with XML-like think blocks
        {
            "name": "Code with think blocks",
            "input": """<think>Let me write a function to calculate factorial</think>
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)"""
        },
        
        # Test Case 4: Code with explanatory text
        {
            "name": "Code with explanations",
            "input": """First, let's create a class for a rectangle.
Here's the implementation:

class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height

You can use it to calculate areas."""
        },
        
        # Test Case 5: Invalid code
        {
            "name": "Invalid code",
            "input": """def broken_function(
    print("This is broken"
)"""
        },
        
        # Test Case 6: Mixed content
        {
            "name": "Mixed content",
            "input": """Let me show you how to create a simple calculator.

<think>I'll create a basic calculator class</think>

```python
class Calculator:
    def add(self, x, y):
        return x + y
    
    def subtract(self, x, y):
        return x - y

# Example usage:
calc = Calculator()
result = calc.add(5, 3)
```

You can use this calculator like shown above."""
        }
    ]
    
    logger.info("=== Starting Code Extraction Tests ===")
    
    for test_case in test_cases:
        logger.info(f"\nTesting: {test_case['name']}")
        logger.info(f"Input:\n{test_case['input']}")
        
        result = model_manager._clean_code_output(test_case['input'])
        
        logger.info(f"Result:\n{result}")
        logger.info("---")
    
    logger.info("\n=== Code Extraction Tests Complete ===")

if __name__ == "__main__":
    test_code_extraction() 