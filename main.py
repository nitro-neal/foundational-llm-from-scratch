from llm_from_scratch import SimpleTokenizerV1
from llm_from_scratch import DataProcessor
from llm_from_scratch import GPTDatasetV1
from llm_from_scratch import SelfAttentionV1
from llm_from_scratch import SelfAttentionV2
from llm_from_scratch import MultiHeadAttention

import torch.nn as nn

from torch.utils.data import DataLoader
import tiktoken
import torch

def create_dataloader_v1(txt, batch_size=4, max_length = 256, stride = 128, shuffle = True, drop_last=True, num_workers = 0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader

if __name__ == "__main__":
    print("\n--LLM From Scratch--\n")

    with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded_text = tokenizer.encode(raw_text)

    vocab_size = 50257
    output_dim = 256
    max_len = 1024
    context_length = max_len


    token_embedding_layer = nn.Embedding(vocab_size, output_dim)
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

    max_length = 4
    dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length)

    for batch in dataloader:
        x, y = batch

        token_embeddings = token_embedding_layer(x)
        pos_embeddings = pos_embedding_layer(torch.arange(max_length))

        input_embeddings = token_embeddings + pos_embeddings

        break

    print(input_embeddings.shape)

    torch.manual_seed(123)

    context_length = max_length
    d_in = output_dim
    d_out = d_in

    mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)

    batch = input_embeddings
    context_vecs = mha(batch)

    print("context_vecs.shape:", context_vecs.shape)

    '''
    urls = ["https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"]
    files = ["data/the-verdict.txt"]

    data_processor = DataProcessor(urls, files)
    data_processor.download()
    vocab = data_processor.create_vocabulary()

    # tokenizer = SimpleTokenizerV1(vocab)
    # Vocab is already hardceded in this tokenizer:
    tokenizer = tiktoken.get_encoding("gpt2")

    with open("./data/the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    max_length = 4
    dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)

    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)

    print("inputs:\n", inputs)
    print("\ntargets:\n", targets)


    vocab_size = 50257
    output_dim = 256

    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    token_embeddings = token_embedding_layer(inputs)
    print(token_embeddings.shape)

    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    print(pos_embeddings.shape)



    # d_in = 4
    # d_out = 8
    # torch.manual_seed(123)
    # sa_v1 = SelfAttentionV1(d_in, d_out)
    # print(sa_v1(inputs))
    # sa_v2 = SelfAttentionV2(d_in, d_out)
    # print(sa_v2(inputs))

    torch.manual_seed(123)
    batch = torch.stack((inputs, inputs), dim=0)
    print(batch.shape) # 2 inputs with 6 tokens each, and each token has embedding dimension 3

    batch_size, context_length, d_in = batch.shape
    d_out = 2
    mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)

    context_vecs = mha(batch)

    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)

    
    attn_scores = torch.empty(6,6)

    for i, x_i in enumerate(inputs):
        for j, x_j in enumerate(inputs):
            attn_scores[i,j] = torch.dot(x_i, x_j)

    print(attn_scores)

    attn_scores = inputs @ inputs.T
    print(attn_scores)
    '''