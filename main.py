from llm_from_scratch import SimpleTokenizerV1
from llm_from_scratch import DataProcessor

if __name__ == "__main__":
    print("\n--LLM From Scratch--\n")

    urls = ["https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"]
    files = ["data/the-verdict.txt"]

    data_processor = DataProcessor(urls, files)
    data_processor.download()

    vocab = data_processor.create_vocabulary()

    simple_tokenizer = SimpleTokenizerV1(vocab)

    encoded = simple_tokenizer.encode("He")
    print(encoded)

    decoded = simple_tokenizer.decode(encoded)
    print(decoded)