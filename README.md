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

## Laymans Terms

We have all training data as files and can load in as raw_text. We split this data into a vocabulary with each token being mapped to a number.

The splitting is done by this line:

```python
re.split(r'([,.:;?*!"()\']|--|\s)', all_raw_text)
```

The | symbol simply means "OR". So, the command splits the text whenever it finds a punctuation mark from the list, OR a double hyphen, OR a whitespace character.

We then add a few more things to our vocabulary:

```
<|endoftext|>: A marker to signify the end of a text.
<|unk|>: A marker for an unknown word that might not be in the vocabulary.
```

we then sort and then map each word to a number

our output will be:
apple -> 0
banana - > 1
the -> 3
...
<|endoftext|> -> 999
<|unk|> -> 1000

now we need to turn tokens into embeddings

First we do a slding winndow approach for the input and target and create tensor arrays:

```
text: "the quick brown fox jumps over the lazy dog"
max_length: 4
stride: 2

# self.input_ids

[
"the quick brown fox",
"brown fox jumps over",
"jumps over the lazy"
]

# self.target_ids

[
"quick brown fox jumps",
"fox jumps over the",
"over the lazy dog"
]
```

now when we run with these params:

```
    dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)

    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)

    print("inputs:\n", inputs)
    print("\ntargets:\n", targets)

```

we get:

```
inputs:
 tensor([[   40,   367,  2885,  1464],
        [ 1807,  3619,   402,   271],
        [10899,  2138,   257,  7026],
        [15632,   438,  2016,   257],
        [  922,  5891,  1576,   438],
        [  568,   340,   373,   645],
        [ 1049,  5975,   284,   502],
        [  284,  3285,   326,    11]])

targets:
 tensor([[  367,  2885,  1464,  1807],
        [ 3619,   402,   271, 10899],
        [ 2138,   257,  7026, 15632],
        [  438,  2016,   257,   922],
        [ 5891,  1576,   438,   568],
        [  340,   373,   645,  1049],
        [ 5975,   284,   502,   284],
        [ 3285,   326,    11,   287]])
```

and if we run with these params:

```
    dataloader = create_dataloader_v1(raw_text, batch_size=2, max_length=6, stride=2, shuffle=False)

```

we get:

````
inputs:
 tensor([[  40,  367, 2885, 1464, 1807, 3619],
        [2885, 1464, 1807, 3619,  402,  271]])

targets:
 tensor([[  367,  2885,  1464,  1807,  3619,   402],
        [ 1464,  1807,  3619,   402,   271, 10899]])
        ```
````

Token Embeddings:
And now what this "means" for a mapping for our embedding layer, if we had 4 vocab words with dimention 3

````
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)

--- Embedding Matrix ---
tensor([ king [-0.2646, -0.2884,  0.0729],
        queen [-0.0886,  0.2010, -0.4930],
        man [ 0.0381, -0.7291,  0.5086],
        woman [ 1.0371, -0.8243,  0.4866]], requires_grad=True)
------------------------
--- Word to Vector Mapping ---
'cat' (ID: 0) --> [-0.2646 -0.2884  0.0729]
'dog' (ID: 1) --> [-0.0886  0.201  -0.493 ]
'fox' (ID: 2) --> [ 0.0381 -0.7291  0.5086]
'fly' (ID: 3) --> [ 1.0371 -0.8243  0.4866]
        ```
````

Then we carete position vectors so it understands the fox at the start is different than fox at the end. There is absolute and relative positional embeddings, open ai uses absolute:

```
Positional Embedding Matrix (Max length of 4 positions, 4 dimensions each)

Position Index

Vector P(index)

0

[0.00, 0.01, 0.02, 0.03]

1

[0.04, 0.05, 0.06, 0.07]

2

[0.08, 0.09, 0.10, 0.11]

3

[0.12, 0.13, 0.14, 0.15]


Export to Sheets
Step 2: The Lookup and Summation
Now, we process the phrase "fox jumps over fox" word by word, grabbing the correct vector from each table and adding them together.

Word in Phrase

Position

Token ID

Looked-up Word Vector V(word)

Looked-up Positional Vector P(index)

Final Vector (Sum)

fox

0

0

[0.1, 0.2, 0.3, 0.4]

[0.00, 0.01, 0.02, 0.03]

[0.10, 0.21, 0.32, 0.43]

jumps

1

1

[0.5, 0.5, 0.5, 0.5]

[0.04, 0.05, 0.06, 0.07]

[0.54, 0.55, 0.56, 0.57]

over

2

2

[0.9, 0.8, 0.7, 0.6]

[0.08, 0.09, 0.10, 0.11]

[0.98, 0.89, 0.80, 0.71]

fox

3

0

[0.1, 0.2, 0.3, 0.4]

[0.12, 0.13, 0.14, 0.15]

[0.22, 0.33, 0.44, 0.55]


Export to Sheets
The Resulting Input
The final tensor that gets fed into the next part of the model is a list of these four unique vectors.

Notice the key takeaway:

The first "fox" (at position 0) becomes [0.10, 0.21, 0.32, 0.43].

The second "fox" (at position 3) becomes [0.22, 0.33, 0.44, 0.55].

Even though they are the same word, their final vector representations are different because the positional information has been successfully encoded.
```

### Chapter 2 Summary:

As part of the input processing pipeline, input text is first broken up into individual tokens. These tokens are then converted into token IDs using a vocabulary. The token IDs are converted into embedding vectors to which positional embeddings of a similar size are added, resulting in input embeddings that are used as input for the main LLM layers.

# Chapter 3

## Simple self attention mechanism.

The goal is to compute a "context vector" for each element.

"your journey starts here" each has a token embedings. journey is [.5,.8,.6]

we compute intermediate value w (attention scores) by computing dot product of journey and all other input tokens. this gives value w for each other attention score. we then normalize so that they all add up to 1. (we use summax so it is percentges and positive).

finally we compute the context vector by multiplying the embedding input token (journy) with attention weights aand then summing the resulting vectors.

## Self attention mechanism with trainable weights

we compute query(q), key (k), and value (v) vectors. The query (q) vector is obtained via matrix multiplication between the input and weight matrix Wq. Similarly we obtain the key and value vectors via matrix multiplication involving the weight matrices Wk and Wv.

Causal Attention (Masking): This is a crucial modification for text-generation models like GPT. It ensures that when calculating attention for a given word, the model can only attend to previous words in the sequence. This is done by "masking" or hiding all future words, forcing the model to be predictive. Training with causal attention results in a decoder-style model.

## Dropout and Masking:

A regularization technique used to prevent overfitting. During training, it randomly sets a fraction of neuron activations to zero (e.g., 10% in real-world models). This forces the model to learn more robust and distributed representations, as it cannot rely on any single neuron. Models trained with dropout generalize better to new, unseen data.

## Multi-head attention

enhances the self-attention mechanism by allowing the model to simultaneously focus on different types of information from multiple perspectives. It works by running several attention "heads" in parallel, where each head has its own independent set of learned weight matrices (Wq, Wk, and Wv). Each head processes the same input sequence but learns to capture different relationships, such as syntactic dependencies or semantic similarities. The individual outputs from all heads are then concatenated and passed through a final linear layer to produce a single, unified output vector. This parallel approach enables the model to build a much richer and more nuanced understanding of the text by integrating various contextual features at once.

the smallest gpt-2 model (117 million parameters) has 12 attention heads and a context vector embedidng size of 768, the largets (1.5 billion parameters) has 25 attention heads and a context vector embedding size of 1600. the embedding sizes of th token inputs and context embeddings are the same in gpt models (d_in = d_out)
