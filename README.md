# LLM
Repository contains LLM built to convert text to SQL using custom BPE with hugging face tokenizer and tiktoken

# Text-to-SQL GPT: Custom BPE vs. Tiktoken

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A minimal, educational implementation of a **Decoder-Only Transformer (GPT)** trained from scratch to translate natural language questions into SQL queries. 

This repository demonstrates two distinct tokenization strategies for training Large Language Models on specific domains:
1.  **Custom BPE:** Training a tokenizer specifically on the dataset.
2.  **Tiktoken:** Using OpenAI's pre-trained tokenizer.

---

## Project Structure

| File | Description |
| :--- | :--- |
| `text_to_sql_generator_with_custom_bpe.py` | **Approach A:** Trains a custom Byte-Pair Encoding (BPE) tokenizer using the Hugging Face `tokenizers` library. This creates a tiny, highly efficient vocabulary (1000 tokens) optimized strictly for SQL syntax. |
| `text_to_sql_generator_with_tiktoken.py` | **Approach B:** Uses OpenAI's `tiktoken` (GPT-2 encoding). This uses a massive general-purpose vocabulary (~50k tokens), demonstrating how pre-trained tokenizers handle domain-specific data. |

---

## Key Features

* **Synthetic Data Generation:** Automatically generates thousands of "Question -> SQL" pairs (e.g., *"Show me the price from products"* $\to$ `SELECT price FROM products;`).
* **Transformer from Scratch:** Implements a full GPT-style architecture in raw PyTorch:
    * Multi-Head Self-Attention
    * Feed-Forward Networks
    * Layer Normalization & Residual Connections
    * Positional Embeddings
* **Custom Tokenizer Training:** Shows how to train a BPE tokenizer on a raw text corpus.
* **Inference Pipeline:** Includes a generation loop with `top-k` sampling (via `torch.multinomial`).

---

## Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/yourusername/text-to-sql-gpt.git](https://github.com/yourusername/text-to-sql-gpt.git)
    cd text-to-sql-gpt
    ```

2.  **Install Dependencies**
    ```bash
    pip install torch tokenizers tiktoken
    ```

---

## Usage

### Option 1: Run with Custom Tokenizer
This script will first generate the data, train a new tokenizer (`custom_sql_tokenizer.json`), and then train the model.

python text_to_sql_generator_with_custom_bpe.py

### Option 2: Run with TiktokenThis script uses the pre-trained gpt2 tokenizer.
python text_to_sql_generator_with_tiktoken.py

Pros: Plug-and-play, no tokenizer training needed.
Cons: The vocabulary is much larger than needed for simple SQL, resulting in a larger model memory footprint.

### Model Architecture
The model is a standard Decoder-Only Transformer , similar to GPT-2/3.

Embeddings: Learned token embeddings + Learnable positional embeddings.

Blocks: 4-6 Layers of MultiheadAttention followed by FeedForward networks.

Attention: Masked Self-Attention (Causal) to prevent the model from "cheating" by seeing future tokens.

Optimization: AdamW optimizer with Cross-Entropy Loss.

### Comparison of Approaches

| Feature | Custom BPE | Tiktoken (GPT-2) |
| :--- | :--- | :--- |
| **Vocab Size** | ~1,000 (Tiny) | ~50,257 (Large) |
| **Embedding Param** | Low (1k * 128) | High (50k * 384) |
| **Token Efficiency** | High (1 token = "SELECT") | Med (Might split "SELECT" if rare) |
| **Setup** | Must train tokenizer | Ready to use |

Input:
"List all users where name is Alice."
SQL:
SELECT * FROM users WHERE name = 'Alice';

