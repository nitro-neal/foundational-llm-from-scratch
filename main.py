from llm_from_scratch import SimpleTokenizerV1
from llm_from_scratch import DataProcessor

import tiktoken

if __name__ == "__main__":
    print("\n--LLM From Scratch--\n")

    urls = ["https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"]
    files = ["data/the-verdict.txt"]

    data_processor = DataProcessor(urls, files)
    data_processor.download()

    vocab = data_processor.create_vocabulary()

    # tokenizer = SimpleTokenizerV1(vocab)

    # Vocab is already hardceded in this tokenizer:
    tokenizer = tiktoken.get_encoding("gpt2")

    # GPT-2 tokenizer uses 50257 tokens (0 to 50256)
    num_tokens = tokenizer.n_vocab

    # List all tokens
    all_tokens = [tokenizer.decode([i]) for i in range(num_tokens)]

    # Optional: Print first N tokens
    for i, token in enumerate(all_tokens[:100]):  # Change 100 to desired number
        print(f"Token ID {i}: {repr(token)}")

        tokens = tokenizer.encode("Akwirw ier", allowed_special={"<|endoftext|>"})
        print(tokens)

    print(tokenizer.decode(tokens))