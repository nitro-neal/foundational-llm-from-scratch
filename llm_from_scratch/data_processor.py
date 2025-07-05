import os
import re
import urllib.request

class DataProcessor:
    def __init__(self, urls: list, output_files: list):
        if len(urls) != len(output_files):
            raise ValueError("urls and output_files must have the same length")

        self.urls = urls
        self.output_files = output_files
    
    def download(self) -> None:
        for i, file in enumerate(self.output_files):
            if not os.path.exists(file):        
                urllib.request.urlretrieve(self.urls[i], file)

    def create_vocabulary(self) -> dict:
        text_chunks = []
        for file in self.output_files:
            with open(file, "r", encoding="utf-8") as f:
                text_chunks.append(f.read())

        all_raw_text = "".join(text_chunks)

        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', all_raw_text)
        preprocessed = [valid_split for item in preprocessed if (valid_split := item.strip())]

        all_words = sorted(list(set(preprocessed)))
        
        vocab = {}
        for i, word in enumerate(all_words):
            vocab[word] = i

        return vocab

        