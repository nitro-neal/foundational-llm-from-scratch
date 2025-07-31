
import os
import time
import tiktoken
import torch
from gpt_model import GPTModel
from dataset_v1 import GPTDatasetV1, DataLoader
from gpt_download import download_and_load_gpt2

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    Generates a sequence of new tokens autoregressively using a trained model.

    This function takes an initial sequence of token IDs and generates a specified
    number of new tokens. It operates in a loop, where in each step, it uses the
    current sequence to predict the very next token. This new token is then
    appended to the sequence, and the process repeats. This implementation uses
    a simple "greedy decoding" strategy, where the most likely token is chosen
    at each step.

    Args:
        model (torch.nn.Module): The trained Transformer-based language model.
            It should accept a tensor of token IDs and return a tensor of logits.

        idx (torch.Tensor): A 2D tensor of token IDs representing the initial
            context or prompt.
            Shape: (batch_size, num_tokens)

        max_new_tokens (int): The number of new tokens to generate and append
            to the input sequence.

        context_size (int): The maximum number of tokens the model can process
            at once (its context window). This is used to truncate the input
            sequence if it grows too long.

    Returns:
        torch.Tensor: The original `idx` tensor with `max_new_tokens` appended
            to the end along the token dimension.
            Shape: (batch_size, num_tokens + max_new_tokens)
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:,-context_size:]
        with torch.no_grad():
            logits = model(idx_cond) # same as model.forward()

        # gets the last row
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())

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

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch) # doing model(input_batch) is same as model.forward(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # Calculate loss gradients
            optimizer.step() # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig("loss-plot.pdf")
    plt.show()


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx

def train_and_save(model_name, model_config, tokenizer):
    GPT_CONFIG_124M = model_config

    with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
        text_data = f.read()

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval();  # Disable dropout during inference

    '''
    start_context = "Every effort moves you"

    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(start_context, tokenizer),
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    '''


    # Train/validation ratio
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]


    torch.manual_seed(123)

    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    
    # Sanity check
    total_characters = len(text_data)
    total_tokens = len(tokenizer.encode(text_data))

    print("Characters:", total_characters)
    print("Tokens:", total_tokens)

    if total_tokens * (train_ratio) < GPT_CONFIG_124M["context_length"]:
        print("Not enough tokens for the training loader. "
            "Try to lower the `GPT_CONFIG_124M['context_length']` or "
            "increase the `training_ratio`")

    if total_tokens * (1-train_ratio) < GPT_CONFIG_124M["context_length"]:
        print("Not enough tokens for the validation loader. "
            "Try to lower the `GPT_CONFIG_124M['context_length']` or "
            "decrease the `training_ratio`")
        

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Note:
    # Uncommenting the following lines will allow the code to run on Apple Silicon chips, if applicable,
    # which is approximately 2x faster than on an Apple CPU (as measured on an M3 MacBook Air).
    # However, the resulting loss values may be slightly different.

    #if torch.cuda.is_available():
    #    device = torch.device("cuda")
    #elif torch.backends.mps.is_available():
    #    device = torch.device("mps")
    #else:
    #    device = torch.device("cpu")
    #
    # print(f"Using {device} device.")

    '''
    model.to(device) # no assignment model = model.to(device) necessary for nn.Module classes

    with torch.no_grad(): # Disable gradient tracking for efficiency because we are not training, yet
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)

    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)
    '''

    start_time = time.time()

    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

    num_epochs = 10
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context="Every effort moves you", tokenizer=tokenizer
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")


    token_ids = generate(
        model=model,
        idx=text_to_token_ids("Every effort moves you", tokenizer),
        max_new_tokens=15,
        context_size=GPT_CONFIG_124M["context_length"],
        top_k=25,
        temperature=1.4
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        }, 
        model_name
    )

    return model

    # epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    # plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
        
    '''
    # --- Story Starters & Literary ---
    generate_and_print_sample(model, tokenizer, device, "Once upon a time, in a kingdom far away,")
    generate_and_print_sample(model, tokenizer, device, "It was a dark and stormy night; the rain fell in torrents—")
    generate_and_print_sample(model, tokenizer, device, "The old house stood on a hill overlooking the town, and its windows looked like")
    generate_and_print_sample(model, tokenizer, device, "In a hole in the ground there lived a")
    generate_and_print_sample(model, tokenizer, device, "The last thing I saw before everything went black was")
    generate_and_print_sample(model, tokenizer, device, "Call me Ishmael.")
    generate_and_print_sample(model, tokenizer, device, "Two roads diverged in a yellow wood, and sorry I could not travel both")
    generate_and_print_sample(model, tokenizer, device, "It was the best of times, it was the worst of times,")
    generate_and_print_sample(model, tokenizer, device, "The mystery began on a Tuesday morning with a single, misplaced")
    generate_and_print_sample(model, tokenizer, device, "For sale: baby shoes,")

    # --- Factual & Informational ---
    generate_and_print_sample(model, tokenizer, device, "The capital of France is")
    generate_and_print_sample(model, tokenizer, device, "Photosynthesis is the process used by plants, algae, and certain bacteria to")
    generate_and_print_sample(model, tokenizer, device, "The theory of relativity was developed by")
    generate_and_print_sample(model, tokenizer, device, "The primary colors are")
    generate_and_print_sample(model, tokenizer, device, "Water is composed of two elements:")
    generate_and_print_sample(model, tokenizer, device, "The internet is a global network of interconnected computers that")
    generate_and_print_sample(model, tokenizer, device, "The human brain is responsible for")
    generate_and_print_sample(model, tokenizer, device, "The Great Wall of China was built to")
    generate_and_print_sample(model, tokenizer, device, "A black hole is a region of spacetime where gravity is so strong that nothing—no particles or even electromagnetic radiation such as light—can")
    generate_and_print_sample(model, tokenizer, device, "The main difference between a democracy and a republic is")

    # --- Common Phrases & Questions ---
    generate_and_print_sample(model, tokenizer, device, "The cat sat on the")
    generate_and_print_sample(model, tokenizer, device, "The quick brown fox jumps over the")
    generate_and_print_sample(model, tokenizer, device, "To be, or not to be, that is the")
    generate_and_print_sample(model, tokenizer, device, "The early bird gets the")
    generate_and_print_sample(model, tokenizer, device, "What is the meaning of life?")
    generate_and_print_sample(model, tokenizer, device, "The best way to learn a new skill is to")
    generate_and_print_sample(model, tokenizer, device, "My favorite thing to do on a rainy day is")
    generate_and_print_sample(model, tokenizer, device, "If I could have any superpower, it would be")
    generate_and_print_sample(model, tokenizer, device, "The secret to happiness is")
    generate_and_print_sample(model, tokenizer, device, "Hello, world! My name is")

    # --- Technical & Code ---
    generate_and_print_sample(model, tokenizer, device, "import numpy as np")
    generate_and_print_sample(model, tokenizer, device, "def calculate_factorial(n):")
    generate_and_print_sample(model, tokenizer, device, "public static void main(String[] args) {")
    generate_and_print_sample(model, tokenizer, device, "SELECT user_id, username FROM users WHERE")
    generate_and_print_sample(model, tokenizer, device, "A function in Python is defined using the `def` keyword, followed by")
    generate_and_print_sample(model, tokenizer, device, "The most common data structures are")
    generate_and_print_sample(model, tokenizer, device, "/* This is a CSS comment. The following rule will change the color of all paragraphs to blue. */\np {")
    generate_and_print_sample(model, tokenizer, device, "console.log('Hello,")
    generate_and_print_sample(model, tokenizer, device, "git commit -m \"")
    generate_and_print_sample(model, tokenizer, device, "<html>\n  <head>\n    <title>")

    # --- Instructions & How-To ---
    generate_and_print_sample(model, tokenizer, device, "To bake a delicious chocolate cake, the first step is to")
    generate_and_print_sample(model, tokenizer, device, "Here is a simple recipe for scrambled eggs:")
    generate_and_print_sample(model, tokenizer, device, "To tie a proper knot, you must first")
    generate_and_print_sample(model, tokenizer, device, "The five steps to building a successful startup are: 1.")
    generate_and_print_sample(model, tokenizer, device, "Before you begin assembling the furniture, make sure you have the following tools:")
    generate_and_print_sample(model, tokenizer, device, "To solve the equation 2x + 5 = 15, you should first")
    generate_and_print_sample(model, tokenizer, device, "The best way to meditate is to find a quiet place, sit comfortably, and")
    generate_and_print_sample(model, tokenizer, device, "A guide to brewing the perfect cup of coffee:")
    generate_and_print_sample(model, tokenizer, device, "To properly care for a houseplant, you need to consider")
    generate_and_print_sample(model, tokenizer, device, "Let's write a story together. You start:")
    '''


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

import numpy as np

def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight, 
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias, 
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, 
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight, 
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias, 
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale, 
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift, 
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale, 
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift, 
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

if __name__ == "__main__":

    # model_name = "my_model_and_optimizer_2.pth"
    model_name = "gpt2"

    GPT_CONFIG_124M = {
        "vocab_size": 50257,   # Vocabulary size
        "context_length": 256, # Shortened context length (orig: 1024)
        "emb_dim": 768,        # Embedding dimension
        "n_heads": 12,         # Number of attention heads
        "n_layers": 12,        # Number of layers
        "drop_rate": 0.1,      # Dropout rate
        "qkv_bias": False      # Query-key-value bias
    }

    tokenizer = tiktoken.get_encoding("gpt2")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Note:
    # Uncommenting the following lines will allow the code to run on Apple Silicon chips, if applicable,
    # which is approximately 2x faster than on an Apple CPU (as measured on an M3 MacBook Air).
    # However, the resulting loss values may be slightly different.

    #if torch.cuda.is_available():
    #    device = torch.device("cuda")
    #elif torch.backends.mps.is_available():
    #    device = torch.device("mps")
    #else:
    #    device = torch.device("cpu")
    #
    # print(f"Using {device} device.")

    model = None
    
    if model_name != "gpt2" and not os.path.exists(model_name):
        print("\nTraining model... will save as: ", model_name)
        model = train_and_save(model_name, GPT_CONFIG_124M, tokenizer)
        print("\n ~~ Finished training ~~ ")

    elif model_name == "gpt2":
        print("\nLoading gpt2 model with name: ", model_name)

        settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")

        # Define model configurations in a dictionary for compactness
        model_configs = {
            "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
            "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
            "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
            "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
        }

        # Copy the base configuration and update with specific model settings
        gpt_model_name = "gpt2-small (124M)"  # Example model name
        # gpt_model_name = "gpt2-xl (1558M)"
        # NEW_CONFIG = GPT_CONFIG_124M.copy()
        GPT_CONFIG_124M.update(model_configs[gpt_model_name])
        GPT_CONFIG_124M.update({"context_length": 1024, "qkv_bias": True})

        model = GPTModel(GPT_CONFIG_124M)
        model.eval();
    
        load_weights_into_gpt(model, params)
        model.to(device);
    
        print("\n ~~ Finished loading ~~ ")
    
    else:
        print("\nLoading model with name: ", model_name)
        checkpoint = torch.load(model_name, weights_only=True)

        model = GPTModel(GPT_CONFIG_124M)
        model.load_state_dict(checkpoint["model_state_dict"])

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.1)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        model.train();
        print("\n ~~ Finished loading ~~ ")


    with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
        text_data = f.read()

    # Train/validation ratio
    train_ratio = 0.70
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

    num_epochs = 10
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context="Every effort moves you", tokenizer=tokenizer
    )

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

    '''
    for i in range(100):

        token_ids = generate(
            model=model,
            idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
            max_new_tokens=25,
            context_size=GPT_CONFIG_124M["context_length"],
            top_k=50,
            temperature=1.5
        )

        print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
    '''


    



    