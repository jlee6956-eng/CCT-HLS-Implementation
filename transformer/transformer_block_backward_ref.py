import math
import numpy as np

# ------------------------------------------------------------
# Reference forward + analytic backward for transformer_block
# using NORMAL backprop (no LoRA): WQ/WK/WV/WO/W1/W2 are all
# trainable and all receive real weight gradients.
#
# Used to generate the expected values in
# transformer_block_backward_tb.cpp.
# ------------------------------------------------------------

rng = np.random.default_rng(0)

TOKENS = 3
EMBED_DIM = 4
NUM_HEADS = 2
HEAD_DIM = EMBED_DIM // NUM_HEADS
MLP_DIM = 6

EPS = 1e-5

_erf = np.frompyfunc(math.erf, 1, 1)

def erf(x):
    return _erf(x).astype(np.float32)

# ------------------------------------------------------------
# Per-token (per-row) primitives and their backward passes.
# ------------------------------------------------------------

def layer_norm(x):
    mean = x.mean(axis=1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=1, keepdims=True)
    std = np.sqrt(var + EPS)
    return (x - mean) / std

def layer_norm_backward(x, y, dy):
    mean = x.mean(axis=1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=1, keepdims=True)
    std = np.sqrt(var + EPS)
    mean_dy = dy.mean(axis=1, keepdims=True)
    mean_dy_y = (dy * y).mean(axis=1, keepdims=True)
    return (dy - mean_dy - y * mean_dy_y) / std

def softmax_rows(x):
    shifted = x - x.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=1, keepdims=True)

def softmax_backward_rows(probs, dprobs):
    dot = (dprobs * probs).sum(axis=1, keepdims=True)
    return probs * (dprobs - dot)

def gelu(x):
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))

def gelu_backward(x, dy):
    cdf = 0.5 * (1.0 + erf(x / np.sqrt(2.0)))
    pdf = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x * x)
    return dy * (cdf + x * pdf)

def attention_scores(Q, K):
    return (Q @ K.T) / np.sqrt(HEAD_DIM)

def attention_scores_backward(Q, K, dScores):
    draw = dScores / np.sqrt(HEAD_DIM)
    return draw @ K, draw.T @ Q

def attention_value_backward(probs, V, dOut):
    return dOut @ V.T, probs.T @ dOut

def mlp_backward_full(X, W1, W2, inner, dOut):
    inner_gelu = gelu(inner)
    d_inner_gelu = dOut @ W2.T
    dW2 = inner_gelu.T @ dOut
    d_inner = gelu_backward(inner, d_inner_gelu)
    dX = d_inner @ W1.T
    dW1 = X.T @ d_inner
    return dX, dW1, dW2

def residual_backward(dOut):
    return dOut, dOut

# ------------------------------------------------------------
# Random weights (all trainable -- no LoRA).
# ------------------------------------------------------------

X = rng.standard_normal((TOKENS, EMBED_DIM)).astype(np.float32)

WQ = rng.standard_normal((EMBED_DIM, EMBED_DIM)).astype(np.float32)
WK = rng.standard_normal((EMBED_DIM, EMBED_DIM)).astype(np.float32)
WV = rng.standard_normal((EMBED_DIM, EMBED_DIM)).astype(np.float32)
WO = rng.standard_normal((EMBED_DIM, EMBED_DIM)).astype(np.float32)

W1 = rng.standard_normal((EMBED_DIM, MLP_DIM)).astype(np.float32)
W2 = rng.standard_normal((MLP_DIM, EMBED_DIM)).astype(np.float32)

# ------------------------------------------------------------
# Forward (caches everything the backward pass needs).
# ------------------------------------------------------------

def forward(X):
    layer1 = layer_norm(X)

    Q = layer1 @ WQ
    K = layer1 @ WK
    V = layer1 @ WV

    concatenated = np.zeros((TOKENS, EMBED_DIM), dtype=np.float32)
    probs_cache, Qh_cache, Kh_cache, Vh_cache = {}, {}, {}, {}

    for h in range(NUM_HEADS):
        s, e = h * HEAD_DIM, (h + 1) * HEAD_DIM
        Qh, Kh, Vh = Q[:, s:e], K[:, s:e], V[:, s:e]
        probs = softmax_rows(attention_scores(Qh, Kh))
        concatenated[:, s:e] = probs @ Vh
        probs_cache[h], Qh_cache[h], Kh_cache[h], Vh_cache[h] = probs, Qh, Kh, Vh

    layer2 = concatenated @ WO
    layer3 = X + layer2
    layer4 = layer_norm(layer3)

    inner = layer4 @ W1
    layer5 = gelu(inner) @ W2

    out = layer3 + layer5

    cache = dict(layer1=layer1, Q=Q, K=K, V=V,
                 probs_cache=probs_cache, Qh_cache=Qh_cache,
                 Kh_cache=Kh_cache, Vh_cache=Vh_cache,
                 concatenated=concatenated,
                 layer3=layer3, layer4=layer4, inner=inner)
    return out, cache

out, cache = forward(X)

# ------------------------------------------------------------
# Analytic backward, mirroring transformer_block_backward_impl.
# ------------------------------------------------------------

def backward(dOut, cache):
    layer1, layer3, layer4, inner, concatenated = (
        cache['layer1'], cache['layer3'], cache['layer4'], cache['inner'],
        cache['concatenated'])

    dLayer3_a, dLayer5 = residual_backward(dOut)
    dLayer4, dW1, dW2 = mlp_backward_full(layer4, W1, W2, inner, dLayer5)
    dLayer3_b = layer_norm_backward(layer3, layer4, dLayer4)
    dLayer3 = dLayer3_a + dLayer3_b

    dX_a, dLayer2 = residual_backward(dLayer3)
    dConcatenated = dLayer2 @ WO.T
    dWO = concatenated.T @ dLayer2

    dQ, dK, dV = (np.zeros_like(cache['Q']), np.zeros_like(cache['K']),
                  np.zeros_like(cache['V']))

    for h in range(NUM_HEADS):
        s, e = h * HEAD_DIM, (h + 1) * HEAD_DIM
        probs = cache['probs_cache'][h]
        Qh, Kh, Vh = cache['Qh_cache'][h], cache['Kh_cache'][h], cache['Vh_cache'][h]

        dProbs, dVh = attention_value_backward(probs, Vh, dConcatenated[:, s:e])
        dScores = softmax_backward_rows(probs, dProbs)
        dQh, dKh = attention_scores_backward(Qh, Kh, dScores)

        dQ[:, s:e], dK[:, s:e], dV[:, s:e] = dQh, dKh, dVh

    dLayer1_q = dQ @ WQ.T
    dWQ = layer1.T @ dQ
    dLayer1_k = dK @ WK.T
    dWK = layer1.T @ dK
    dLayer1_v = dV @ WV.T
    dWV = layer1.T @ dV

    dLayer1 = dLayer1_q + dLayer1_k + dLayer1_v
    dX_b = layer_norm_backward(X, layer1, dLayer1)

    return dX_a + dX_b, dWQ, dWK, dWV, dWO, dW1, dW2

dOut_seed = rng.standard_normal((TOKENS, EMBED_DIM)).astype(np.float32)
dX, dWQ, dWK, dWV, dWO, dW1, dW2 = backward(dOut_seed, cache)

# ------------------------------------------------------------
# Finite-difference sanity checks (independent of the analytic
# formulas above -- catches chain-rule mistakes).
# ------------------------------------------------------------

def directional_loss(Xp, WQp, WKp, WVp, WOp, W1p, W2p):
    global WQ, WK, WV, WO, W1, W2
    WQ_orig, WK_orig, WV_orig, WO_orig, W1_orig, W2_orig = WQ, WK, WV, WO, W1, W2
    WQ, WK, WV, WO, W1, W2 = WQp, WKp, WVp, WOp, W1p, W2p
    o, _ = forward(Xp)
    WQ, WK, WV, WO, W1, W2 = WQ_orig, WK_orig, WV_orig, WO_orig, W1_orig, W2_orig
    return np.sum(o * dOut_seed)

h = 1e-3

def fd_grad(param, dparam_analytic, name):
    fd = np.zeros_like(param, dtype=np.float64)
    it = np.nditer(param, flags=['multi_index'])
    for _ in it:
        idx = it.multi_index
        orig = param[idx]
        param[idx] = orig + h
        lp = directional_loss(X, WQ, WK, WV, WO, W1, W2)
        param[idx] = orig - h
        lm = directional_loss(X, WQ, WK, WV, WO, W1, W2)
        param[idx] = orig
        fd[idx] = (lp - lm) / (2 * h)
    print(f"max |analytic {name} - finite-diff {name}| =",
          np.max(np.abs(dparam_analytic - fd)))

fd_grad(X, dX, "dX")
fd_grad(WQ, dWQ, "dWQ")
fd_grad(WK, dWK, "dWK")
fd_grad(WV, dWV, "dWV")
fd_grad(WO, dWO, "dWO")
fd_grad(W1, dW1, "dW1")
fd_grad(W2, dW2, "dW2")

# ------------------------------------------------------------
# Flattened C arrays for the HLS testbench.
# ------------------------------------------------------------

def dump(name, arr):
    body = ", ".join(f"{v:.9f}f" for v in arr.flatten())
    print(f"float {name}[{arr.size}] = {{{body}}};")

print(f"\n// TOKENS={TOKENS} EMBED_DIM={EMBED_DIM} NUM_HEADS={NUM_HEADS} "
      f"MLP_DIM={MLP_DIM}")
for name, arr in [
    ("X", X), ("WQ", WQ), ("WK", WK), ("WV", WV), ("WO", WO),
    ("W1", W1), ("W2", W2), ("dOut_seed", dOut_seed),
    ("expected_out", out), ("expected_dX", dX),
    ("expected_dWQ", dWQ), ("expected_dWK", dWK),
    ("expected_dWV", dWV), ("expected_dWO", dWO),
    ("expected_dW1", dW1), ("expected_dW2", dW2),
]:
    dump(name, arr)
