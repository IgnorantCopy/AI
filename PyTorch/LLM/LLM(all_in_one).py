import os
import requests
import math
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 4
context_size = 64
d_model = 512
num_blocks = 8
num_heads = 4
learning_rate = 1e-3
dropout = 0.1
max_iters = 5000  # total of training iterations <- change this to smaller number for testing
eval_interval = 50  # how often to evaluate
eval_iters = 20  # number of iterations to average for evaluation
device = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_SEED = 1337
torch.manual_seed(TORCH_SEED)

# download the dataset
filename = 'data/sales_textbook.txt'
if not os.path.exists(filename):
    url = 'https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling/resolve/main/sales_textbook.txt?download=true'
    with open(filename, 'w') as f:
        f.write(requests.get(url).text)

with open(filename, 'r', encoding='utf-8') as f:
    text = f.read()

# tokenize the text
encoding = tiktoken.get_encoding('cl100k_base')
tokenized_text = encoding.encode(text)
max_token_value = max(tokenized_text) + 1
tokenized_text = torch.tensor(tokenized_text, dtype=torch.long, device=device)

# split into train and validation
train_size = int(len(tokenized_text) * 0.9)
train_data = tokenized_text[:train_size]
valid_data = tokenized_text[train_size:]


class FeedforwardNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ffn(x)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_model // num_heads, bias=False)
        self.Wk = nn.Linear(d_model, d_model // num_heads, bias=False)
        self.Wv = nn.Linear(d_model, d_model // num_heads, bias=False)
        # apply mask
        self.register_buffer('mask', torch.tril(torch.ones(context_size, context_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, T, C = x.shape
        assert T <= context_size
        assert C == d_model
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        attention_scores = (Q @ K.transpose(-2, -1) / math.sqrt(K.size(-1)))
        attention_scores = attention_scores.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        attention_scores = F.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)
        return attention_scores @ V


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([ScaledDotProductAttention() for _ in range(num_heads)])
        self.projection_layer = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = torch.cat([head(x) for head in self.heads], dim=-1)
        output = self.projection_layer(output)
        output = self.dropout(output)
        return output


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.multi_head_attention = MultiHeadAttention()
        self.feedforward_network = FeedforwardNetwork()

    def forward(self, x):
        x = x + self.multi_head_attention(self.layer_norm1(x))
        x = x + self.feedforward_network(self.layer_norm2(x))
        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_lookup_table = nn.Embedding(max_token_value + 1, d_model)
        self.transformer_blocks = nn.Sequential(*(
            [TransformerBlock() for _ in range(num_blocks)] +
            [nn.LayerNorm(d_model)]
        ))
        self.model_out_linear_layer = nn.Linear(d_model, max_token_value)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        position_encoding_lookup_table = torch.zeros(context_size, d_model, device=device)
        position = torch.arange(0, context_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        position_encoding_lookup_table[:, 0::2] = torch.sin(position * div_term)
        position_encoding_lookup_table[:, 1::2] = torch.cos(position * div_term)
        position_embedding = position_encoding_lookup_table[:T, :].to(device)
        x = self.token_embedding_lookup_table(idx) + position_embedding
        x = self.transformer_blocks(x)
        logits = self.model_out_linear_layer(x)

        if targets is not None:
            B, T, C = logits.shape
            logits_reshaped = logits.view(B * T, C)
            targets_reshaped = targets.view(B * T)
            loss = F.cross_entropy(input=logits_reshaped, target=targets_reshaped)
        else:
            loss = None
        return logits, loss

    def generate(self, idx, max_new_tokens=100):
        for _ in range(max_new_tokens):
            # Crop idx to the max size of the positional embedding table
            idx_crop = idx[:, -context_size:]
            # Get prediction
            logits, _ = self.forward(idx_crop)
            # Get the last time step from logits where dimensions of the logits are (B, T, C)
            logits_last_timestep = logits[:, -1, :]
            # Apply softmax to get probabilities
            probs = F.softmax(input=logits_last_timestep, dim=-1)
            # Sample from the probabilities' distribution
            idx_next = torch.multinomial(input=probs, num_samples=1)
            # Append the sampled indexes index_next to idx
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = Model().to(device)


def get_batch(split: str):
    data = train_data if split == 'train' else valid_data
    idxs = torch.randint(low=0, high=len(data) - context_size, size=(batch_size,))
    x = torch.stack([data[idx:idx + context_size] for idx in idxs]).to(device)
    y = torch.stack([data[idx + 1:idx + context_size + 1] for idx in idxs]).to(device)
    return x, y


# calculate the loss
@torch.no_grad()
def estimate_loss():
    output = {}
    model.eval()
    for split in ['train', 'valid']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x_batch, y_batch = get_batch(split)
            _, loss = model(x_batch, y_batch)
            losses[k] = loss.item()
        output[split] = losses.mean()
    model.train()
    return output


# create the optimizer
optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)
tracked_losses = list()
for step in range(max_iters):
    if step % eval_iters == 0 or step == max_iters - 1:
        losses = estimate_loss()
        tracked_losses.append(losses)
        print('Step:', step, 'Train Loss:', round(losses['train'].item(), 3), 'Valid Loss:', 'Validation Loss:',
              round(losses['valid'].item(), 3))

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Save the model
torch.save(model.state_dict(), 'model.pt')

# Evaluate the model
model.eval()
start = "Steve Jobs is"
start_ids = encoding.encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
y = model.generate(x, max_new_tokens=100)
print('----------------------------')
print(encoding.decode(y[0].tolist()))
print('----------------------------')
