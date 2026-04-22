import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
#  PrunableLinear (copied from Step 1 — keep everything in one
#  file per submission requirement, we'll merge at the end)
# ─────────────────────────────────────────────────────────────
class PrunableLinear(nn.Module):
    """
    Custom Linear layer with learnable gate_scores.
    gates = sigmoid(gate_scores) ∈ (0,1)
    pruned_weight = weight * gates
    output = F.linear(x, pruned_weight, bias)
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        self.weight      = nn.Parameter(torch.empty(out_features, in_features))
        self.bias        = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        nn.init.zeros_(self.gate_scores)   # sigmoid(0) = 0.5 → half-open start

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates         = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_scores).detach()

    def sparsity(self, threshold: float = 1e-2) -> float:
        return (self.get_gates() < threshold).float().mean().item()

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


# ─────────────────────────────────────────────────────────────
#  SelfPruningNet  — CIFAR-10 Classifier
# ─────────────────────────────────────────────────────────────
class SelfPruningNet(nn.Module):
    """
    Feed-forward neural network for CIFAR-10 image classification.

    Architecture:
        Input : 32 × 32 × 3 = 3072 raw pixels (flattened)
        FC1   : PrunableLinear(3072 → 1024)  + BatchNorm + ReLU + Dropout
        FC2   : PrunableLinear(1024 → 512)   + BatchNorm + ReLU + Dropout
        FC3   : PrunableLinear(512  → 256)   + BatchNorm + ReLU + Dropout
        FC4   : PrunableLinear(256  → 10)    → logits (no activation)

    Why this architecture?
      - 4 prunable layers give the sparsity loss enough gates to work with
      - BatchNorm stabilizes training (important when gates change the
        effective weight scale during training)
      - Dropout + sparsity loss together prevent overfitting
      - Final layer outputs raw logits for CrossEntropyLoss

    The network exposes all PrunableLinear layers via self.prunable_layers
    so that compute_sparsity_loss() can iterate over them cleanly.
    """

    def __init__(self, dropout_rate: float = 0.3):
        super().__init__()

        # ── Prunable fully-connected layers ──────────────────────
        self.fc1 = PrunableLinear(3072, 1024)
        self.fc2 = PrunableLinear(1024, 512)
        self.fc3 = PrunableLinear(512,  256)
        self.fc4 = PrunableLinear(256,  10)

        # ── Batch normalization (applied after each prunable layer) ─
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)

        # ── Dropout for regularization ───────────────────────────
        self.dropout = nn.Dropout(p=dropout_rate)

        # ── Convenience list for sparsity loss computation ───────
        self.prunable_layers = [self.fc1, self.fc2, self.fc3, self.fc4]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 3, 32, 32) — standard CIFAR-10 tensor
        returns: (batch, 10) — class logits
        """
        # Flatten image: (batch, 3, 32, 32) → (batch, 3072)
        x = x.view(x.size(0), -1)

        # FC1
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # FC2
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # FC3
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)

        # FC4 — output logits (no activation, CrossEntropyLoss handles it)
        x = self.fc4(x)
        return x

    def compute_sparsity_loss(self) -> torch.Tensor:
        """
        Sparsity Loss = L1 norm of ALL gate values across ALL PrunableLinear layers.

        Why L1?  The L1 norm (sum of absolute values) is known to push values
        toward exactly zero (creates a sparse solution), unlike L2 which only
        shrinks them toward zero. Since gates = sigmoid(scores) are always
        positive, L1 norm = simple sum of all gate values.

        Returns a scalar tensor (differentiable — gradients flow back to
        gate_scores through the sigmoid).
        """
        all_gates = []
        for layer in self.prunable_layers:
            gates = torch.sigmoid(layer.gate_scores)   # shape: (out, in)
            all_gates.append(gates.view(-1))            # flatten to 1D

        all_gates_concat = torch.cat(all_gates)         # all gates in one vector
        return all_gates_concat.sum()                   # L1 norm (all positive)

    def overall_sparsity(self, threshold: float = 1e-2) -> float:
        """Returns the fraction of all gates below `threshold` across all layers."""
        total_gates   = 0
        pruned_gates  = 0
        for layer in self.prunable_layers:
            gates         = layer.get_gates()
            total_gates  += gates.numel()
            pruned_gates += (gates < threshold).sum().item()
        return pruned_gates / total_gates

    def layer_sparsity_report(self, threshold: float = 1e-2) -> dict:
        """Returns per-layer sparsity for detailed reporting."""
        return {
            f"fc{i+1}": layer.sparsity(threshold)
            for i, layer in enumerate(self.prunable_layers)
        }


# ─────────────────────────────────────────────────────────────
#  Verification Tests
# ─────────────────────────────────────────────────────────────
def run_tests():
    print("=" * 60)
    print("  STEP 2: SelfPruningNet — Verification Tests")
    print("=" * 60)

    all_passed = True

    # ── Test 1: Forward pass on CIFAR-10 sized batch ────────────
    print("\n[Test 1] Forward pass with CIFAR-10 sized input...")
    model = SelfPruningNet()
    model.eval()

    batch_size  = 32
    dummy_imgs  = torch.randn(batch_size, 3, 32, 32)  # (B, C, H, W)
    with torch.no_grad():
        logits = model(dummy_imgs)

    expected = (batch_size, 10)
    passed   = logits.shape == expected
    all_passed &= passed
    print(f"  Input shape  : {tuple(dummy_imgs.shape)}")
    print(f"  Output shape : {tuple(logits.shape)}")
    print(f"  Expected     : {expected}")
    print(f"  Result       : {'✅ PASS' if passed else '❌ FAIL'}")

    # ── Test 2: Sparsity loss is a valid scalar & differentiable ─
    print("\n[Test 2] Sparsity loss — scalar & differentiable...")
    model     = SelfPruningNet()
    model.train()

    dummy_imgs = torch.randn(16, 3, 32, 32)
    logits     = model(dummy_imgs)
    labels     = torch.randint(0, 10, (16,))

    clf_loss      = F.cross_entropy(logits, labels)
    sparsity_loss = model.compute_sparsity_loss()
    lam           = 1e-4
    total_loss    = clf_loss + lam * sparsity_loss

    is_scalar    = sparsity_loss.dim() == 0
    is_positive  = sparsity_loss.item() > 0
    total_loss.backward()
    grad_exists  = model.fc1.gate_scores.grad is not None

    passed = is_scalar and is_positive and grad_exists
    all_passed &= passed
    print(f"  Classification loss : {clf_loss.item():.4f}")
    print(f"  Sparsity loss       : {sparsity_loss.item():.4f}")
    print(f"  Total loss          : {total_loss.item():.4f}")
    print(f"  Is scalar           : {'✅ Yes' if is_scalar else '❌ No'}")
    print(f"  Is positive         : {'✅ Yes' if is_positive else '❌ No'}")
    print(f"  Grad flows to gates : {'✅ Yes' if grad_exists else '❌ No'}")
    print(f"  Result              : {'✅ PASS' if passed else '❌ FAIL'}")

    # ── Test 3: Total parameter count is reasonable ──────────────
    print("\n[Test 3] Parameter count breakdown...")
    model        = SelfPruningNet()
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Should be > 5M (lots of gates) but < 50M (not absurd)
    passed = 5_000_000 < total_params < 50_000_000
    all_passed &= passed

    # Per-layer breakdown
    layer_info = {
        "fc1 (3072→1024)": model.fc1,
        "fc2 (1024→512)" : model.fc2,
        "fc3 (512→256)"  : model.fc3,
        "fc4 (256→10)"   : model.fc4,
    }
    for name, layer in layer_info.items():
        cnt = sum(p.numel() for p in layer.parameters())
        print(f"  {name}: {cnt:>10,} params (weight + bias + gate_scores)")

    print(f"  {'─'*48}")
    print(f"  Total params    : {total_params:>10,}")
    print(f"  Trainable params: {trainable:>10,}")
    print(f"  Result          : {'✅ PASS' if passed else '❌ FAIL'}")

    # ── Test 4: Sparsity reporting works ────────────────────────
    print("\n[Test 4] Sparsity reporting (initial — gates start at 0.5)...")
    model   = SelfPruningNet()
    report  = model.layer_sparsity_report()
    overall = model.overall_sparsity()

    # At init gate_scores=0 → sigmoid(0)=0.5 → nothing pruned yet
    # So sparsity should be 0% initially (no gates below 1e-2 threshold)
    passed  = overall == 0.0
    all_passed &= passed
    for name, sparsity in report.items():
        print(f"  {name} sparsity : {sparsity*100:.1f}%")
    print(f"  Overall sparsity : {overall*100:.1f}%  (expected 0% at init)")
    print(f"  Result           : {'✅ PASS' if passed else '❌ FAIL'}")

    # ── Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if all_passed:
        print("  🎉 ALL TESTS PASSED — SelfPruningNet is ready!")
        print("  ✅ Ready to move to Step 3: Training loop + sparsity loss.")
    else:
        print("  ❌ SOME TESTS FAILED — Please check output above.")
    print("=" * 60)
    print()


if __name__ == "__main__":
    run_tests()