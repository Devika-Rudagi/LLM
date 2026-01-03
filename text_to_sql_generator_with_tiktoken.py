import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import random

# --- 1. CONFIGURATION ---
batch_size = 16
block_size = 128       # Context window
max_iters = 5000       # Training steps
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 100
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

torch.manual_seed(1337)

# --- 2. DATA GENERATION (Synthetic SQL) ---
def generate_sql_data(num_samples=5000):
    tables = ['users', 'products', 'orders', 'employees']
    columns = {
        'users': ['id', 'name', 'email', 'signup_date'],
        'products': ['id', 'title', 'price', 'stock_count'],
        'orders': ['id', 'user_id', 'order_date', 'total_amount'],
        'employees': ['id', 'first_name', 'department', 'salary']
    }
    templates = [
        ("Show me the {col} from {tab}.", "SELECT {col} FROM {tab};"),
        ("List all {tab} where {col} is {val}.", "SELECT * FROM {tab} WHERE {col} = '{val}';"),
        ("Find the {col} for {tab} with id {val}.", "SELECT {col} FROM {tab} WHERE id = {val};"),
        ("Count the number of {tab}.", "SELECT count(*) FROM {tab};")
    ]

    data = []
    for _ in range(num_samples):
        tab = random.choice(tables)
        col = random.choice(columns[tab])
        val = random.randint(1, 100)
        tmpl_q, tmpl_s = random.choice(templates)

        q = tmpl_q.format(tab=tab, col=col, val=val)
        s = tmpl_s.format(tab=tab, col=col, val=val)

        # We add a special delimiter <END> so the model knows when to stop
        entry = f"Question: {q}\nSQL: {s}\n<END>\n"
        data.append(entry)
    return "".join(data)

raw_text = generate_sql_data()

"""# create token and model definitions"""

# --- 3. TOKENIZATION (BPE with tiktoken) ---
# We use the GPT-4 tokenizer ('cl100k_base') for 100K tokens and gpt2 for 50k
enc = tiktoken.get_encoding("gpt2")

# Encode the entire dataset
train_ids = enc.encode(raw_text)
print(f"Total tokens in dataset: {len(train_ids)}")
print(f"Vocabulary size: {enc.n_vocab}")

# Convert to tensor
data = torch.tensor(train_ids, dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Because BPE vocab is large (100k+), we need to update vocab_size
vocab_size = enc.n_vocab

# --- 4. MODEL COMPONENTS (Standard Transformer) ---

def get_batch(split):
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

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        # Using PyTorch's optimized MultiheadAttention
        self.sa = nn.MultiheadAttention(embed_dim=n_embd, num_heads=n_head,
                                        dropout=dropout, batch_first=True)
        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        B, T, C = x.shape
        # 1. MANUALLY CREATE MASK
        # Generates a (T, T) matrix of -inf (top right) and 0 (bottom left).
        # This prevents the model from seeing future tokens.
        attn_mask = nn.Transformer.generate_square_subsequent_mask(T).to(device)
        x_norm = self.ln1(x)
        # We pass x_norm three times (Query, Key, Value).
        # We pass 'attn_mask' to force causality.
        attn_out, _ = self.sa(x_norm, x_norm, x_norm, attn_mask=attn_mask, need_weights=False)
        x = x + attn_out
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

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
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

"""#Training"""

# --- 5. TRAINING LOOP ---
model = GPTLanguageModel()
m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"Model parameters: {sum(p.numel() for p in m.parameters())/1e6:.2f}M")
print("Starting training...")

for iter in range(3000): #max_iters
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: Train Loss {losses['train']:.4f}, Val Loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

"""#Testing"""

# --- 6. INFERENCE (Text-to-SQL) ---
print("\n--- INFERENCE TEST ---")

def generate_sql_bpe(question):
    prompt = f"Question: {question}\nSQL:"
    # Encode with BPE
    input_ids = enc.encode(prompt)
    context = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    # Generate
    output_ids = m.generate(context, max_new_tokens=50)[0].tolist()
    # Decode with BPE
    output_text = enc.decode(output_ids)

    # Parse output
    try:
        start_marker = "SQL:"
        end_marker = "<END>"

        # We start searching AFTER the prompt to avoid finding the prompt's own "SQL:"
        start_idx = output_text.find(start_marker) + len(start_marker)

        # Extract everything after "SQL:"
        generated_sql = output_text[start_idx:]

        # Stop at <END> or newline if <END> isn't generated
        if end_marker in generated_sql:
            generated_sql = generated_sql.split(end_marker)[0]
        else:
             generated_sql = generated_sql.split('\n')[0]

        return generated_sql.strip()
    except:
        return output_text

# Test Questions
test_qs = [
    "Show me the price from products.",
    "List all users where name is Alice.",
    "Count the number of employees.",
    "Find the department for employees with id 10."
]

for q in test_qs:
    print(f"Q: {q}")
    print(f"SQL: {generate_sql_bpe(q)}")
    print("-" * 30)
