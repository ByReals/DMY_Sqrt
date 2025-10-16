import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt

# === Configuration (must match training) ===
class Config:
    model_name = "DMYsqrt"
    block_size = 30
    vocab = ['<PAD>', '<EOS>'] + list("0123456789+=")
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

def format_test_input(a: int, b: int, max_digits=4):
    """Format input string padded with zeros to width = max_digits"""
    # e.g. if max_digits = 4, 58 -> "0058", so input is "0058+0064="
    sa = str(a).rjust(max_digits, '0')
    sb = str(b).rjust(max_digits, '0')
    return f"{sa}+{sb}="

def prepare_input_tensor(input_str: str):
    arr = encode(input_str, add_eos=False)
    if len(arr) < cfg.block_size:
        arr += [pad_idx] * (cfg.block_size - len(arr))
    else:
        arr = arr[:cfg.block_size]
    return torch.tensor(arr, dtype=torch.long).unsqueeze(0)

def test_model(model, num_tests=1000, max_digits=4, print_samples=5):
    model.eval()
    correct = 0
    results = []
    with torch.no_grad():
        for _ in range(num_tests):
            a = random.randint(0, 10**max_digits - 1)
            b = random.randint(0, 10**max_digits - 1)
            inp = format_test_input(a, b, max_digits)
            truth = str(a + b)
            x = prepare_input_tensor(inp).to(cfg.device)
            logits = model(x)
            pred_ids = torch.argmax(logits, dim=-1)[0].cpu().tolist()
            pred = decode(pred_ids)
            is_correct = (pred == truth)
            if len(results) < print_samples:
                results.append((inp, pred, truth, is_correct))
            if is_correct:
                correct += 1
    accuracy = correct / num_tests * 100
    print(f"Tested {num_tests} additions (max_digits={max_digits}) → Accuracy: {accuracy:.2f}%")
    for inp, pred, truth, ok in results:
        print(f"Input:     {inp}")
        print(f"Predicted: {pred}")
        print(f"Target:    {truth}")
        print(f"Correct:   {ok}")
        print("-----")
    return accuracy

def accuracy_vs_digits(model, max_digits_list=[1,2,3,4,5], tests_per_digit=1000):
    accuracies = []
    for d in max_digits_list:
        acc = test_model(model, num_tests=tests_per_digit, max_digits=d, print_samples=0)
        accuracies.append(acc)
    plt.figure(figsize=(8,5))
    plt.plot(max_digits_list, accuracies, marker='o')
    plt.title("Accuracy vs Max Digits")
    plt.xlabel("Max digits in operands")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    # Save the plot
    plt.savefig("accuracy_vs_digits.png")
    print("Saved accuracy_vs_digits.png")
    return accuracies

if __name__ == "__main__":

    print("Loading model from:", cfg.save_path)
    model = DMYsqrtModel().to(cfg.device)
    model.load_state_dict(torch.load(cfg.save_path, map_location=cfg.device))
    model.eval()

    # Test on random inputs
    test_model(model, num_tests=100, max_digits=4, print_samples=5)

    # Plot how accuracy changes with input size
    accuracy_vs_digits(model, max_digits_list=[1,2,3,4,5], tests_per_digit=500)
