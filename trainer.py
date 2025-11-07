import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast

import random

# ======================================
# CONFIGURATION
# ======================================
class Config:
    model_name = "DMY_MathTrainer"
    block_size = 30
    batch_size = 128
    epochs = 30
    vocab = ['<PAD>', '<EOS>'] + list("0123456789+-×÷=")  # includes all math ops
    embedding_dim = 256
    n_layers = 8
    n_heads = 8
    dropout = 0.1
    lr = 5e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_path = f"{model_name}.pt"
    grad_clip = 1.0

cfg = Config()

# ======================================
# TOKENIZATION HELPERS
# ======================================
stoi = {ch: i for i, ch in enumerate(cfg.vocab)}
itos = {i: ch for ch, i in stoi.items()}
pad_idx = stoi['<PAD>']
eos_idx = stoi['<EOS>']
vocab_size = len(cfg.vocab)

def encode(s, add_eos=False):
    """Convert string to list of token indices."""
    arr = [stoi[c] for c in s if c in stoi]
    if add_eos:
        arr.append(eos_idx)
    return arr

def decode(toks):
    """Convert token indices back to string."""
    chars = []
    for i in toks:
        if i == eos_idx:
            break
        if i == pad_idx:
            continue
        chars.append(itos.get(i, ''))
    return ''.join(chars)

# ======================================
# DATASET CLASS
# ======================================
class MathDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_str, output_str = self.data[idx]
        x = encode(input_str, add_eos=False)
        y = encode(output_str, add_eos=True)

        # pad or truncate to block size
        if len(x) < cfg.block_size:
            x += [pad_idx] * (cfg.block_size - len(x))
        else:
            x = x[:cfg.block_size]

        if len(y) < cfg.block_size:
            y += [pad_idx] * (cfg.block_size - len(y))
        else:
            y = y[:cfg.block_size]

        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# ======================================
# MODEL
# ======================================
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

# ======================================
# EVALUATION
# ======================================
def evaluate(model, loader):
    model.eval()
    correct_seq, total_seq = 0, 0
    correct_tokens, total_tokens = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            logits = model(x)
            preds = torch.argmax(logits, dim=-1)

            for pred_ids, target_ids in zip(preds, y):
                pred_str = decode(pred_ids.cpu().tolist())
                true_str = decode(target_ids.cpu().tolist())
                total_seq += 1
                if pred_str == true_str:
                    correct_seq += 1

                for p, t in zip(pred_ids.cpu().tolist(), target_ids.cpu().tolist()):
                    if t == pad_idx:
                        continue
                    total_tokens += 1
                    if p == t:
                        correct_tokens += 1

    seq_acc = correct_seq / total_seq if total_seq > 0 else 0.0
    token_acc = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    return seq_acc, token_acc

# ======================================
# TRAINING LOOP
# ======================================
def train(model, train_loader, val_loader):
    model.to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)
    scaler = GradScaler()

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            optimizer.zero_grad()
            with autocast(device_type=cfg.device):
                logits = model(x)
                loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        scheduler.step()
        val_seq_acc, val_token_acc = evaluate(model, val_loader)

        print(f"[{cfg.model_name}] Epoch {epoch+1}/{cfg.epochs} | Loss: {total_loss:.4f} | Val Seq Acc: {val_seq_acc:.4f} | Val Token Acc: {val_token_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # show a few predictions
        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(cfg.device)
                preds = torch.argmax(model(x), dim=-1)
                for i in range(min(3, x.size(0))):
                    print("Input:     ", decode(x[i].cpu().tolist()))
                    print("Predicted: ", decode(preds[i].cpu().tolist()))
                    print("Target:    ", decode(y[i].cpu().tolist()))
                    print("---")
                break

        torch.save(model.state_dict(), cfg.save_path)

# ======================================
# LOAD DATA FROM problems.txt
# ======================================
def load_dataset_from_txt(filepath="problems.txt"):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    raw_data = []
    for line in lines:
        line = line.strip()
        if '=' not in line:
            continue
        input_part, output_part = line.split('=')
        input_str = input_part + '='   # e.g. "3+4="
        output_str = output_part.strip()  # e.g. "7"
        raw_data.append((input_str, output_str))

    random.shuffle(raw_data)
    split = int(0.9 * len(raw_data))
    train_set = MathDataset(raw_data[:split])
    val_set = MathDataset(raw_data[split:])
    return train_set, val_set

# ======================================
# MAIN
# ======================================
if __name__ == "__main__":
    print(f"Training model: {cfg.model_name} on device: {cfg.device}")
    train_data, val_data = load_dataset_from_txt("problems.txt")
    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=cfg.batch_size, num_workers=4, pin_memory=True)
    model = DMYsqrtModel()
    train(model, train_loader, val_loader)
    print(f"Model {cfg.model_name} saved to {cfg.save_path}")
