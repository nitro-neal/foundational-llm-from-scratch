# LLM From Scratch

A simple yet robust LLM tokenizer implementation built from scratch for educational and practical purposes. This project demonstrates fundamental concepts in natural language processing, specifically text tokenization and vocabulary building.

## 🚀 Features

- **Simple Text Tokenization**: Vocabulary-based encoding and decoding of text
- **Data Processing**: Automatic downloading and processing of text data from URLs
- **Robust Error Handling**: Comprehensive error handling and validation
- **Type Safety**: Full type hints for better code reliability
- **Comprehensive Testing**: Extensive test suite with high coverage
- **Production Ready**: Follows Python best practices and coding standards

## 📦 Installation

### Using pip (recommended)
```bash
pip install llm-from-scratch
```

### From source
```bash
git clone https://github.com/yourusername/llm-from-scratch.git
cd llm-from-scratch
pip install -e .
```

### Development installation
```bash
git clone https://github.com/yourusername/llm-from-scratch.git
cd llm-from-scratch
pip install -e ".[dev]"
```

## 🎯 Quick Start

### Basic Usage

```python
from llm_from_scratch import SimpleTokenizerV1, DataProcessor

# 1. Download and process training data
urls = ["https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"]
files = ["data/the-verdict.txt"]

processor = DataProcessor(urls, files)
processor.download()

# 2. Create vocabulary from the text
vocab = processor.create_vocabulary()

# 3. Initialize tokenizer
tokenizer = SimpleTokenizerV1(vocab)

# 4. Encode text to tokens
text = "Hello world! This is a test."
tokens = tokenizer.encode(text)
print(f"Tokens: {tokens}")

# 5. Decode tokens back to text
decoded_text = tokenizer.decode(tokens)
print(f"Decoded: {decoded_text}")
```

### Advanced Usage

```python
from llm_from_scratch import DataProcessor, SimpleTokenizerV1

# Process multiple data sources
urls = [
    "https://example.com/text1.txt",
    "https://example.com/text2.txt"
]
files = ["data/text1.txt", "data/text2.txt"]

processor = DataProcessor(urls, files)

# Force re-download even if files exist
processor.download(force=True)

# Create vocabulary with minimum frequency filtering
vocab = processor.create_vocabulary(min_frequency=2)

# Initialize tokenizer
tokenizer = SimpleTokenizerV1(vocab)

# Batch processing
texts = [
    "First example text.",
    "Second example text!",
    "Third example with punctuation?"
]

for text in texts:
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    print(f"Original: {text}")
    print(f"Tokens: {tokens}")
    print(f"Decoded: {decoded}")
    print("-" * 50)
```

## 🏗️ Project Structure

