"""
Integer-only forward pass for the quantized Fashion-MNIST MLP (ABACUS-8).

Pure NumPy, no PyTorch -- this is the golden reference the RTL testbench
(ABACUS-9 onward) is validated against bit-for-bit.

Every accumulation stage is explicitly cast to np.int32/np.int64. NumPy
does not pick a safe wide accumulator on its own: int8 @ int8 stays int8
and silently wraps on overflow -- verified directly:
    np.array([100,100,100], dtype=np.int8) @ np.array([100,100,100], dtype=np.int8)
    -> 48, not 30000 (wrapped mod 256, not promoted to a safe width)
Casting inputs to int32 before each matmul makes the accumulator width an
explicit, auditable choice matching the 32-bit accumulator ABACUS-7's
requantization targets, not an accident of whatever NumPy happens to pick.

Consumes model/quant_params_ptq.npz (ABACUS-6) and
model/requant_constants.npz (ABACUS-7).
"""

import numpy as np

LAYERS = ("fc1", "fc2", "fc3")


def quantize_input(x, s_x, qmin=-128, qmax=127):
    """Float -> int8 given a step scale s_x (see ABACUS-6)."""
    q = np.round(x / s_x)
    return np.clip(q, qmin, qmax).astype(np.int8)


def linear_int32(x_int8, w_int8, b_int32):
    """int8 x int8 matmul, explicitly accumulated in int32, plus int32 bias."""
    x32 = x_int8.astype(np.int32)
    w32 = w_int8.astype(np.int32)
    acc = x32 @ w32.T
    assert acc.dtype == np.int32, "accumulator silently widened -- check numpy promotion rules"
    return (acc + b_int32.astype(np.int32)).astype(np.int32)


def requantize(acc_int32, M0, shift, qmin=-128, qmax=127, apply_relu=True):
    """
    Rescale an int32 accumulator by M = M0 * 2**(shift-31) using only
    integer multiply/add/shift (the M decomposition from ABACUS-7), then
    fuse ReLU into the clamp before truncating to int8.

    acc_int32 * M0 can exceed int32 (e.g. a ~24-bit accumulator times a
    31-bit M0 needs ~55 bits), so the intermediate is deliberately widened
    to int64 for this one step -- mirroring a wider multiply-accumulate
    register in hardware (e.g. a DSP48E2's 48-bit output), not a NumPy
    convenience.
    """
    acc64 = acc_int32.astype(np.int64)
    total_shift = 31 - shift
    if total_shift > 0:
        rounding = np.int64(1) << np.int64(total_shift - 1)
        rescaled = (acc64 * np.int64(M0) + rounding) >> np.int64(total_shift)
    else:
        rescaled = acc64 * np.int64(M0) << np.int64(-total_shift)
    lo = 0 if apply_relu else qmin
    return np.clip(rescaled, lo, qmax).astype(np.int8)


def forward(x_float, params):
    """
    x_float: (N, 784) normalized float32/float64 pixels.
    params:  dict from load_params(), below.
    Returns int32 logits, shape (N, 10). fc3's output is never
    requantized: argmax over a single positive per-tensor-scaled int32
    accumulator is scale-invariant (see ABACUS-7), so classification
    needs no dequantization at all.
    """
    q0 = quantize_input(x_float, params["fc1_s_x"])

    acc1 = linear_int32(q0, params["fc1_weight_int8"], params["fc1_bias_int32"])
    q1 = requantize(acc1, params["fc1_M0"], params["fc1_shift"])

    acc2 = linear_int32(q1, params["fc2_weight_int8"], params["fc2_bias_int32"])
    q2 = requantize(acc2, params["fc2_M0"], params["fc2_shift"])

    acc3 = linear_int32(q2, params["fc3_weight_int8"], params["fc3_bias_int32"])
    return acc3


def load_params(quant_path="quant_params_ptq.npz", requant_path="requant_constants.npz"):
    q = np.load(quant_path)
    r = np.load(requant_path)
    params = {}
    for name in LAYERS:
        params[f"{name}_weight_int8"] = q[f"{name}_weight_int8"]
        params[f"{name}_bias_int32"] = q[f"{name}_bias_int32"]
    params["fc1_s_x"] = float(q["fc1_s_x"])
    for name in ("fc1", "fc2"):
        params[f"{name}_M0"] = int(r[f"{name}_M0"])
        params[f"{name}_shift"] = int(r[f"{name}_shift"])
    return params


if __name__ == "__main__":
    # Smoke test: run the golden model over the real Fashion-MNIST test
    # set and report accuracy. (Full logging/analysis is ABACUS-10; this
    # is just "does the integer-only path actually work" for ABACUS-8.)
    from torchvision import datasets, transforms

    MEAN, STD = 0.2860, 0.3530
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MEAN,), (STD,)),
        transforms.Lambda(lambda x: x.view(-1)),
    ])
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tfm)

    x_all = np.stack([np.asarray(img) for img, _ in test_set]).astype(np.float32)
    y_all = np.array([label for _, label in test_set])

    params = load_params()
    logits = forward(x_all, params)
    preds = logits.argmax(axis=1)
    acc = (preds == y_all).mean()
    print(f"golden model (pure NumPy, integer-only) test acc: {acc*100:.2f}%")
