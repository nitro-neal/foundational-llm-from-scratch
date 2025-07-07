import re

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
                                
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]

        new_preprocessed = []

        for item in preprocessed:
            if item in self.str_to_int:
                new_preprocessed.append(item)
            else:
                new_preprocessed.append("<|unk|>")

        preprocessed = new_preprocessed

        '''
        preprocessed = [
            item if item in self.str_to_int
            else "<|unk|>" for item in preprocessed
         ]
        '''

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text