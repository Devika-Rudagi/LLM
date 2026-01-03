import torch
import torch.nn as nn
from torch.nn import functional as F
import random
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# --- 1. CONFIGURATION ---
batch_size = 64
block_size = 64        # SQL queries are short, 64 is plenty
max_iters = 3000
eval_interval = 500
learning_rate = 1e-3   # Slightly higher LR for smaller models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 100
n_embd = 128           # Reduced dim since vocab is small
n_head = 4
n_layer = 4
dropout = 0.2
vocab_size = 1000      # Target vocabulary size (Custom!)

torch.manual_seed(1337)

"""# create SQL Data Data

"""

import random

def generate_sql_data(num_samples= 10000):
    # 1. Define simple schema and templates
    tables = ['users', 'products', 'orders', 'employees']
    columns = {
        'users': ['id', 'name', 'email', 'signup_date'],
        'products': ['id', 'title', 'price', 'stock_count'],
        'orders': ['id', 'user_id', 'order_date', 'total_amount'],
        'employees': ['id', 'first_name', 'department', 'salary']
    }

    # 2. Define patterns (English -> SQL)
    templates = [
        (
            "Show me the {col} from {tab}.",
            "SELECT {col} FROM {tab};"
        ),
        (
            "List all {tab} where {col} is {val}.",
            "SELECT * FROM {tab} WHERE {col} = '{val}';"
        ),
        (
            "Find the {col} for {tab} with id {val}.",
            "SELECT {col} FROM {tab} WHERE id = {val};"
        ),
        (
            "Count the number of {tab}.",
            "SELECT count(*) FROM {tab};"
        )
    ]

    # 3. Generate 10,000 examples
    data = ""
    for _ in range(10000):
        # Pick a random table and column
        tab = random.choice(tables)
        col = random.choice(columns[tab])
        val = random.randint(1, 100)

        # Pick a random question template
        tmpl_q, tmpl_s = random.choice(templates)

        # Fill in the blanks
        q = tmpl_q.format(tab=tab, col=col, val=val)
        s = tmpl_s.format(tab=tab, col=col, val=val)

        # Format: Question -> SQL -> <END>
        data += f"Question: {q}\nSQL: {s}\n<END>\n"
    return "\n".join(data)

# 4. Save to file
raw_text = generate_sql_data()
with open("sql_dataset.txt", "w") as f:
    f.write(raw_text)

print("Data generated.")

"""We will use the Hugging Face tokenizers library to train a custom BPE tokenizer specifically for your SQL dataset."""

!pip install tokenizers

"""#Train the tokenizer - Custom BPE tokenizer
Instead of building stoi and itos manually, we feed our sql_data.txt into the BPE trainer. It will find the most common patterns (like "SELECT", "count", "id") and assign them unique IDs.
"""

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

print("Training Custom BPE Tokenizer...")

# 1. Initialize an empty BPE tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace() # Split by whitespace first

# 2. Configure the trainer
# vocab_size=1000 is plenty for our tiny SQL vocabulary.
# In GPT-4, this is usually 100,000+.
trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[END]"], vocab_size=1000)

# 3. Train on your file
files = ["sql_data.txt"]
tokenizer.train(files, trainer)

# 4. Save it
tokenizer.save("custom_sql_tokenizer.json")

# --- TEST IT ---
encoded = tokenizer.encode("SELECT * FROM users")
print(f"Tokens: {encoded.tokens}")
print(f"IDs:    {encoded.ids}")
# You should see something like: ['SELECT', '*', 'FROM', 'users']
# and IDs like [12, 5, 14, 25] (Single integers for whole words!)

"""#Prepare data"""

# Encode the whole dataset
full_ids = tokenizer.encode(raw_text).ids
data_tensor = torch.tensor(full_ids, dtype=torch.long)

# Train/Val split
n = int(0.9 * len(data_tensor))
train_data = data_tensor[:n]
val_data = data_tensor[n:]

# Update vocab_size to exactly what was trained
actual_vocab_size = tokenizer.get_vocab_size()

"""# Model Definitions (same as miniGPT)"""

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
        # Creates a (T, T) matrix where future positions are -inf
        #Generate the mask manually to avoid version errors
        attn_mask = nn.Transformer.generate_square_subsequent_mask(T).to(device)

        x_norm = self.ln1(x)
        attn_out, _ = self.sa(x_norm, x_norm, x_norm, attn_mask = attn_mask,
                              need_weights=False)
        x = x + attn_out
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Note: We use actual_vocab_size here
        self.token_embedding_table = nn.Embedding(actual_vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, actual_vocab_size)

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

model = GPTLanguageModel()
m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"Model parameters: {sum(p.numel() for p in m.parameters())/1e6:.2f}M")
print("Starting training...")

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: Train Loss {losses['train']:.4f}, Val Loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

"""#Test"""

def generate_sql_custom(question):
    prompt = f"Question: {question}\nSQL:"
    input_ids = tokenizer.encode(prompt).ids
    context = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    output_ids = m.generate(context, max_new_tokens=30)[0].tolist()
    output_text = tokenizer.decode(output_ids)

    try:
        start_idx = output_text.find("SQL:") + len("SQL:")
        generated = output_text[start_idx:]
        if "<END>" in generated:
            generated = generated.split("<END>")[0]
        return generated.strip()
    except:
        return output_text

test_qs = [
    "Show me the price from products.",
    "List all users where name is Alice.",
    "Count the number of employees.",
    "Find the department for employees with id 10."
]

for q in test_qs:
    print(f"Q: {q}")
    print(f"SQL: {generate_sql_custom(q)}")
    print("-" * 30)
