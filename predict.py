import torch
import torch.nn as nn

# === Configuration (must match training) ===
class Config:
    model_name = "DMY_MathTrainer"
    block_size = 30
    vocab = ['<PAD>', '<EOS>'] + list("0123456789+-×÷=")
    embedding_dim = 256
    n_layers = 8
    n_heads = 8
    dropout = 0.1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_path = f"{model_name}.pt"

cfg = Config()

stoi = {ch: i for i, ch in enumerate(cfg.vocab)}
itos = {i: ch for ch, i in stoi.items()}
pad_idx = stoi['<PAD>']
eos_idx = stoi['<EOS>']
vocab_size = len(cfg.vocab)

def encode(s, add_eos=False):
    arr = [stoi[c] for c in s]
    if add_eos:
        arr.append(eos_idx)
    return arr

def decode(toks):
    chars = []
    for i in toks:
        if i == eos_idx:
            break
        if i == pad_idx:
            continue
        chars.append(itos.get(i, ''))
    return ''.join(chars)

class DMYsqrtModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, cfg.embedding_dim, padding_idx=pad_idx)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.block_size, cfg.embedding_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.embedding_dim,
            nhead=cfg.n_heads,
            dim_feedforward=4 * cfg.embedding_dim,
            dropout=cfg.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
        self.ln = nn.LayerNorm(cfg.embedding_dim)
        self.head = nn.Linear(cfg.embedding_dim, vocab_size)

    def forward(self, idx):
        tok_emb = self.token_emb(idx)
        x = tok_emb + self.pos_emb[:, :idx.size(1), :]
        x = self.transformer(x)
        x = self.ln(x)
        logits = self.head(x)
        return logits

def prepare_input_tensor(input_str: str):
    arr = encode(input_str, add_eos=False)
    if len(arr) < cfg.block_size:
        arr += [pad_idx] * (cfg.block_size - len(arr))
    else:
        arr = arr[:cfg.block_size]
    return torch.tensor(arr, dtype=torch.long).unsqueeze(0)

def predict_sum(model, input_str):
    if not input_str.endswith('='):
        input_str += '='
    x = prepare_input_tensor(input_str).to(cfg.device)
    model.eval()
    with torch.no_grad():
        logits = model(x)  # shape: (1, block_size, vocab_size)
        pred_ids = torch.argmax(logits, dim=-1)[0].cpu().tolist()
    decoded = decode(pred_ids)
    # Extract the predicted sum after '=' sign
    if '=' in decoded:
        result = decoded.split('=')[1]
    else:
        result = decoded
    return result.strip()

if __name__ == "__main__":
    model = DMYsqrtModel().to(cfg.device)
    model.load_state_dict(torch.load(cfg.save_path, map_location=cfg.device))
    print("Model ready. Type additions like '1234+5678=' and hit Enter (Ctrl+C to exit).")

    while True:
        try:
            user_input = input("Enter addition (e.g., 1234+5678=): ").strip()
            if not user_input:
                continue
            prediction = predict_sum(model, user_input)
            print(f"Predicted result: {prediction}")
        except KeyboardInterrupt:
            print("\nExiting.")
            break
