"""
Self-Pruning Neural Network — FULLY CORRECTED
==============================================

ROOT CAUSE OF 0% SPARSITY (diagnosed via debug output):
  gate_scores were initialized to zeros → every gate started identical
  → sparsity gradient pushed ALL gates equally → no differentiation
  → gates moved in lockstep, never diverged toward 0

FIXES APPLIED:
  1. gate_scores init: zeros → torch.randn * 0.01  (breaks symmetry)
  2. Removed weight_decay from Adam (it was competing with gate gradients)
  3. Higher learning rate for gate_scores via param groups
  4. Lambda values chosen to create visible low/medium/high pruning
  5. More epochs (30) to give gates time to diverge
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import json, time

# ─────────────────────────────────────────────────────────────
# 1.  PrunableLinear  (THE FIX IS HERE: random gate_scores init)
# ─────────────────────────────────────────────────────────────
class PrunableLinear(nn.Module):
    """
    Custom linear layer with learnable gates.

    Forward:
        gates         = sigmoid(gate_scores)          # (out, in) in (0,1)
        pruned_weights = weight * gates               # element-wise
        output        = pruned_weights @ x.T + bias
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        self.weight      = nn.Parameter(torch.empty(out_features, in_features))
        self.bias        = nn.Parameter(torch.zeros(out_features))

        # ★ FIX: small random init breaks symmetry so gates diverge
        self.gate_scores = nn.Parameter(torch.randn(out_features, in_features) * 0.01)

        nn.init.kaiming_uniform_(self.weight, a=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates          = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

    def sparsity(self, threshold: float = 1e-2) -> float:
        with torch.no_grad():
            gates = torch.sigmoid(self.gate_scores)
            return (gates < threshold).float().mean().item()

    def gate_values(self) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.gate_scores).flatten().cpu()


# ─────────────────────────────────────────────────────────────
# 2.  Network
# ─────────────────────────────────────────────────────────────
class SelfPruningNet(nn.Module):
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.fc1 = PrunableLinear(3072, 1024)
        self.fc2 = PrunableLinear(1024, 512)
        self.fc3 = PrunableLinear(512,  256)
        self.fc4 = PrunableLinear(256,  10)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.drop(F.relu(self.bn1(self.fc1(x))))
        x = self.drop(F.relu(self.bn2(self.fc2(x))))
        x = self.drop(F.relu(self.bn3(self.fc3(x))))
        return self.fc4(x)

    def sparsity_loss(self) -> torch.Tensor:
        """Raw L1 sum of all gate values (as specified in case study)."""
        total = torch.tensor(0.0)
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                total = total + torch.sigmoid(m.gate_scores).sum()
        return total

    def overall_sparsity(self, threshold: float = 1e-2) -> float:
        vals = [m.sparsity(threshold) for m in self.modules()
                if isinstance(m, PrunableLinear)]
        return sum(vals) / len(vals) if vals else 0.0

    def per_layer_sparsity(self, threshold: float = 1e-2) -> dict:
        return {
            f"fc{i+1}": round(m.sparsity(threshold) * 100, 2)
            for i, m in enumerate([self.fc1, self.fc2, self.fc3, self.fc4])
        }

    def all_gate_values(self) -> torch.Tensor:
        return torch.cat([m.gate_values() for m in self.modules()
                          if isinstance(m, PrunableLinear)])


# ─────────────────────────────────────────────────────────────
# 3.  Data
# ─────────────────────────────────────────────────────────────
def get_loaders(batch_size: int = 128):
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
    ])
    train_ds = torchvision.datasets.CIFAR10("./data", train=True,  download=True, transform=train_tf)
    test_ds  = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=test_tf)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0),
            DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0))


# ─────────────────────────────────────────────────────────────
# 4.  Train one model
# ─────────────────────────────────────────────────────────────
def train_model(lam: float, train_loader, test_loader,
                device, epochs: int = 30):

    model = SelfPruningNet(dropout=0.3).to(device)

    # ★ FIX: separate param groups — gates get 5× higher lr
    gate_params  = [p for n, p in model.named_parameters() if "gate_scores" in n]
    other_params = [p for n, p in model.named_parameters() if "gate_scores" not in n]

    optimizer = torch.optim.Adam([
        {"params": other_params, "lr": 1e-3},
        {"params": gate_params,  "lr": 5e-3},   # gates need stronger push
    ], weight_decay=0)                            # no weight_decay (competes with gates)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ce = nn.CrossEntropyLoss()

    print(f"\n{'='*65}")
    print(f"  Training with λ = {lam}")
    print(f"{'='*65}")

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        tot_clf = tot_spar = tot_tot = 0.0
        nb = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits    = model(imgs)
            loss_clf  = ce(logits, labels)
            loss_spar = model.sparsity_loss()
            loss_tot  = loss_clf + lam * loss_spar
            loss_tot.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tot_clf  += loss_clf.item()
            tot_spar += loss_spar.item()
            tot_tot  += loss_tot.item()
            nb += 1

        scheduler.step()
        sp = model.overall_sparsity() * 100

        # Show gate score range so we can confirm divergence
        with torch.no_grad():
            gs = model.fc1.gate_scores
            gs_min = gs.min().item()
            gs_max = gs.max().item()

        print(f"  Epoch [{epoch:02d}/{epochs}] "
              f"clf={tot_clf/nb:.4f}  "
              f"spar={tot_spar/nb:.0f}  "
              f"total={tot_tot/nb:.4f}  "
              f"sparsity={sp:.1f}%  "
              f"score_range=[{gs_min:.2f},{gs_max:.2f}]  "
              f"({time.time()-t0:.1f}s)")

    # ── Evaluate ──────────────────────────────────────────────
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            correct += (model(imgs).argmax(1) == labels).sum().item()
            total   += labels.size(0)

    acc      = correct / total * 100
    sparsity = model.overall_sparsity() * 100
    per_layer = model.per_layer_sparsity()

    print(f"\n  ── Final Results (λ={lam}) ──")
    print(f"  Test Accuracy    : {acc:.2f}%")
    print(f"  Overall Sparsity : {sparsity:.2f}%")
    print(f"  Per-layer        : {per_layer}")

    # Save gate values for plotting in Step 4
    gate_vals = model.all_gate_values().numpy().tolist()
    torch.save(model.state_dict(), f"model_lam_{lam}.pth")
    print(f"  Saved → model_lam_{lam}.pth")

    return {
        "lambda":    lam,
        "accuracy":  round(acc, 2),
        "sparsity":  round(sparsity, 2),
        "per_layer": per_layer,
        "gate_values": gate_vals,
    }


# ─────────────────────────────────────────────────────────────
# 5.  Main
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device : {device}")

    train_loader, test_loader = get_loaders(128)
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Test  batches : {len(test_loader)}")

    # Low / Medium / High pruning pressure
    lambdas = [1e-5, 1e-4, 1e-3]

    results = []
    for lam in lambdas:
        r = train_model(lam, train_loader, test_loader, device, epochs=30)
        results.append(r)

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n  Saved → results.json")

    print(f"\n{'='*65}")
    print(f"{'FINAL SUMMARY TABLE':^65}")
    print(f"{'='*65}")
    print(f"  {'Lambda':<12}  {'Test Acc':>10}  {'Sparsity':>10}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}")
    for r in results:
        print(f"  {r['lambda']:<12}  {r['accuracy']:>9.2f}%  {r['sparsity']:>9.2f}%")
    print(f"{'='*65}")
    print("\n  ✅ Done! Run step4_plots.py next.")