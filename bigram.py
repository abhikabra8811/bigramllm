from itertools import islice

import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 10
dropout = 0.2
# ------------

torch.manual_seed(1337)

with open('elon_tweets.txt', 'r', encoding='utf-8') as f:
    text = ''.join(islice(f, 100_000))

#all the unique characters in the text
chars = sorted(list(set(text))) # all unique characters in the text
vocab_size = len(chars) # size of the vocabulary

# create mappings from characters to integers and vice versa
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s:[stoi[c] for c in s] # convert string to list of integers representing characters
decode = lambda l: ''.join([itos[i] for i in l]) # convert list of integers back to string

# Train and test data split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

#class for single head of attention of head_size
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        T = x.size(1)
        q = self.query(x) # [B x T x head_size]
        k = self.key(x) # [B x T x head_size]
        v = self.value(x) # [B x T x head_size]
        # compute attention weights 
        # scale the attention weights by the square root of the key dimension
        # why do we scale with the square root of the key dimension? 
        # This helps to prevent the dot products from growing too large, 
        # which can make the softmax function produce very small gradients.
        wei = q @ k.transpose(-2, -1) * (k.size(-1) ** -0.5) # [B x T x T] attention weights before masking and softmax
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        out = wei @ v # [B x T x head_size] after applying attention weights
        out = self.dropout(out)
        return out

# Multi-head self-attention mechanism
# Instead of having a single attention head, we use multiple heads to capture different aspects of the input.
# This allows the model to learn different types of relationships and dependencies in the input sequence.
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, num_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_head)])
        self.proj = nn.Linear(n_embd, n_embd) # projection layer after concatenating heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # h(x) for each attention head
        # h(x) [B x T x head_size] for each attention head
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # out [B x T x (head_size * num_head)] after concatenating all attention heads
        out = self.proj(out)
        out = self.dropout(out)
        # out [B x T x n_embd] after projection
        return out

class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# block is made of a self-attention layer followed by a feed-forward layer
# It also includes residual connections around each sub-layer.
# It also has layer normalization (if implemented) before each sub-layer.
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadSelfAttention(n_head, head_size)
        self.feed_forward = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd) # layer norm before self-attention
        self.ln2 = nn.LayerNorm(n_embd) # layer norm before feed-forward layer

    def forward(self, x):
        # adding residual connections with layer normalization
        # x [B x T x n_embd] before self-attention
        x = x + self.sa(self.ln1(x))
        # x [B x T x n_embd] after self-attention and residual connection
        x = x + self.feed_forward(self.ln2(x))
        # x [B x T x n_embd] after feed-forward and residual connection
        return x

class GPTModel(nn.Module):
    def __init__(self):
        super().__init__()
        # define the GPT model architecture here
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        #position embedding table for the input sequence
        #it provides a unique embedding for each position in the input sequence
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        #define the transformer blocks
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer normalization before output
        self.lm_head = nn.Linear(n_embd, vocab_size) # language model head for output predictions

        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        #targets B x T
        #idx B x T
        tok_emb = self.token_embedding_table(idx) # token embeddings
        # tok_emb B x T x n_embd
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # position embeddings
        # pos_emb B x T x n_embd
        x = tok_emb + pos_emb
        # x [B x T x n_embd]
        x = self.blocks(x)
        # x [B x T x n_embd] after transformer blocks
        x = self.ln_f(x)
        logits = self.lm_head(x)
        # logits [B x T x vocab_size]

        if targets is None:
            loss = None
        else:
            B, T, VocabSize = logits.shape
            logits = logits.view(B*T, VocabSize)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is the current context of shape (B, T)
        for _ in range(max_new_tokens):
            # get last batch size from the context
            B, T = idx.shape
            # crop context if needed
            idx_cond = idx[:, -block_size:]
            # forward the model to get the logits
            logits, _ = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = GPTModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        loss_values = estimate_loss()
        print(f"Step {iter}: train loss {loss_values['train']:.4f}, val loss {loss_values['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')
    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#generate some text after training
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=1400)
print(generated)