from generative_models.gpt import GPTModel
from generative_models.gpt.tokenizers import CharacterLevelTokenizer

import torch
import torch.optim
from tqdm import tqdm
import sys
import pickle

def causal_batch(token_data, context_size, batch_size):
    n = len(token_data)
    offsets = torch.randint(n - context_size, (batch_size, ))
    arange = torch.arange(context_size)
    idx =  offsets[:, None] + arange[None, :]
    idx, next_idx = torch.clamp(idx, min=0, max=n-1), torch.clamp(idx + 1, min=0, max=n-1)
    x = torch.take(token_data, idx) 
    y = torch.take(token_data, next_idx)
    return x, y

if __name__ == "__main__":
    device = "xpu" if torch.xpu.is_available() else "cpu"
    with open("training_data/odysee.txt", "r", encoding="utf-8") as f:
        text = f.read()
    tokenizer = CharacterLevelTokenizer(text)
    with open("tokenizer.pk", "wb") as ftok:
        pickle.dump(tokenizer, ftok)
    data = tokenizer.encode(text, eof=True)

    context_len = 256
    n_parallel, n_attn = 4,64
    n_embed = n_parallel * n_attn
    num_gpt_blocks = 10
    model = GPTModel(tokenizer.vocab_size, context_len, n_embed, n_attn, num_blocks = num_gpt_blocks).to(device)
    
    lr, epochs, batch_size = 1e-3, 2500, 64
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    pbar = tqdm(range(epochs))
    try:
        for epoch in pbar:
            optimizer.zero_grad()
            batch_x, batch_y = causal_batch(data, context_len, batch_size)
            batch_x, batch_y  = batch_x.to(device), batch_y.to(device)
            _, loss = model(batch_x, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value = 0.1)
            optimizer.step()
            pbar.set_description(f"loss: {loss.item()}")
            if epoch > 0 and epoch % 100 == 0:
                torch.save(model, "gpt.pt")
    finally:
        torch.save(model, "gpt.pt")
