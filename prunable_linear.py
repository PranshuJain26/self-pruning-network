import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
#  PrunableLinear Layer
# ─────────────────────────────────────────────
class PrunableLinear(nn.Module):
    """
    A custom linear layer with learnable 'gate_scores'.

    For each weight w_ij, there is a corresponding gate_score g_ij.
    During the forward pass:
        gates       = sigmoid(gate_scores)      → values in (0, 1)
        pruned_w    = weight * gates            → element-wise multiply
        output      = input @ pruned_w.T + bias → standard linear op

    When a gate approaches 0, the corresponding weight is effectively
    "pruned" (removed) from the network. The L1 sparsity loss (added
    during training) pushes gate values toward exactly 0.

    Args:
        in_features  (int): Size of each input sample
        out_features (int): Size of each output sample
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.in_features  = in_features
        self.out_features = out_features

        # Standard weight & bias — same as nn.Linear
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        self.bias = nn.Parameter(
            torch.zeros(out_features)
        )

        # ⭐ Gate scores — same shape as weight
        # Registered as a Parameter so the optimizer updates it too
        self.gate_scores = nn.Parameter(
            torch.empty(out_features, in_features)
        )

        # Initialize weights with kaiming uniform (same as nn.Linear default)
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

        # Initialize gate_scores near 0 → sigmoid(0) = 0.5
        # This means gates start at ~0.5 (half-open) and learn from there
        nn.init.zeros_(self.gate_scores)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
            1. Transform gate_scores → gates via Sigmoid (ensures 0–1 range)
            2. Compute pruned_weights = weight ⊙ gates  (element-wise)
            3. Return F.linear(x, pruned_weights, bias)
        """
        # Step 1: gate_scores → gates ∈ (0, 1)
        gates = torch.sigmoid(self.gate_scores)

        # Step 2: element-wise multiply weight with gates
        pruned_weights = self.weight * gates

        # Step 3: standard linear transformation
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Returns current gate values (after sigmoid). Useful for analysis."""
        return torch.sigmoid(self.gate_scores).detach()

    def sparsity(self, threshold: float = 1e-2) -> float:
        """Returns fraction of gates below `threshold` (i.e., pruned)."""
        gates = self.get_gates()
        return (gates < threshold).float().mean().item()

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


# ─────────────────────────────────────────────
#  Verification Tests
# ─────────────────────────────────────────────
def run_tests():
    print("=" * 60)
    print("  STEP 1: PrunableLinear — Verification Tests")
    print("=" * 60)

    all_passed = True

    # ── Test 1: Forward pass produces correct output shape ──────────
    print("\n[Test 1] Forward pass output shape...")
    layer = PrunableLinear(in_features=8, out_features=4)
    x     = torch.randn(16, 8)          # batch of 16 samples, 8 features each
    out   = layer(x)

    expected_shape = (16, 4)
    passed = out.shape == expected_shape
    all_passed &= passed
    print(f"  Input shape       : {tuple(x.shape)}")
    print(f"  Output shape      : {tuple(out.shape)}")
    print(f"  Expected shape    : {expected_shape}")
    print(f"  Result            : {'✅ PASS' if passed else '❌ FAIL'}")

    # ── Test 2: Gradients flow through BOTH weight and gate_scores ──
    print("\n[Test 2] Gradient flow through weight and gate_scores...")
    layer     = PrunableLinear(in_features=8, out_features=4)
    x         = torch.randn(16, 8)
    out       = layer(x)
    loss      = out.sum()               # simple scalar loss
    loss.backward()

    weight_grad_exists     = layer.weight.grad is not None
    gate_grad_exists       = layer.gate_scores.grad is not None
    weight_grad_nonzero    = weight_grad_exists and layer.weight.grad.abs().sum().item() > 0
    gate_grad_nonzero      = gate_grad_exists  and layer.gate_scores.grad.abs().sum().item() > 0

    passed = weight_grad_nonzero and gate_grad_nonzero
    all_passed &= passed
    print(f"  weight.grad exists & nonzero     : {'✅ Yes' if weight_grad_nonzero else '❌ No'}")
    print(f"  gate_scores.grad exists & nonzero: {'✅ Yes' if gate_grad_nonzero else '❌ No'}")
    print(f"  Result                           : {'✅ PASS' if passed else '❌ FAIL'}")

    # ── Test 3: Gates are strictly in (0, 1) ────────────────────────
    print("\n[Test 3] Gates are in (0, 1) range after sigmoid...")
    layer = PrunableLinear(in_features=32, out_features=16)
    # Randomize gate_scores to a wide range to stress-test sigmoid
    with torch.no_grad():
        layer.gate_scores.uniform_(-10, 10)

    gates   = layer.get_gates()
    in_range = (gates >= 0).all() and (gates <= 1).all()
    passed   = in_range.item()
    all_passed &= passed
    print(f"  Gate min  : {gates.min().item():.6f}")
    print(f"  Gate max  : {gates.max().item():.6f}")
    print(f"  All in [0,1]: {'✅ Yes' if passed else '❌ No'}")
    print(f"  Result    : {'✅ PASS' if passed else '❌ FAIL'}")

    # ── Test 4: Heavily negative gate_scores → gates ≈ 0 → pruned ──
    print("\n[Test 4] Very negative gate_scores → weights effectively pruned...")
    layer = PrunableLinear(in_features=8, out_features=4)
    with torch.no_grad():
        layer.gate_scores.fill_(-20.0)   # sigmoid(-20) ≈ 0

    gates           = layer.get_gates()
    pruned_weights  = layer.weight * gates
    avg_gate        = gates.mean().item()
    avg_pruned_w    = pruned_weights.abs().mean().item()
    sparsity_pct    = layer.sparsity(threshold=1e-2) * 100

    passed = avg_gate < 0.01 and sparsity_pct == 100.0
    all_passed &= passed
    print(f"  Average gate value         : {avg_gate:.8f}  (should be ≈ 0)")
    print(f"  Average |pruned weight|    : {avg_pruned_w:.8f}  (should be ≈ 0)")
    print(f"  Sparsity (gates < 1e-2)    : {sparsity_pct:.1f}%  (should be 100%)")
    print(f"  Result                     : {'✅ PASS' if passed else '❌ FAIL'}")

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if all_passed:
        print("  🎉 ALL TESTS PASSED — PrunableLinear is working correctly!")
        print("  ✅ Ready to move to Step 2: Build the full network.")
    else:
        print("  ❌ SOME TESTS FAILED — Please check the output above.")
    print("=" * 60)

    # ── Extra info: parameter count ─────────────────────────────────
    print("\n[Info] PrunableLinear(128 → 64) parameter breakdown:")
    demo = PrunableLinear(128, 64)
    for name, param in demo.named_parameters():
        print(f"  {name:20s} | shape: {tuple(param.shape)} | "
              f"count: {param.numel():,}")
    total = sum(p.numel() for p in demo.parameters())
    print(f"  {'TOTAL':20s} | count: {total:,}")
    print()


if __name__ == "__main__":
    run_tests()