import math
import numpy as np

# ------------------------------------------------------------
# Reference forward + analytic backward for transformer_block_lora,
# used to generate the expected values in
# transformer_block_lora_backward_tb.cpp.
#
# Only lora_AQ/lora_BQ/lora_AV/lora_BV are trainable; WQ/WK/WV/WO/
# W1/W2 are frozen, so their backward paths never produce a weight
# gradient -- only a dIn used to keep the chain rule flowing back
# to X (needed when this block is stacked under other LoRA blocks).
# ------------------------------------------------------------

rng = np.random.default_rng(0)

TOKENS = 3
EMBED_DIM = 4
NUM_HEADS = 2
HEAD_DIM = EMBED_DIM // NUM_HEADS
LORA_RANK = 2
MLP_DIM = 6
ALPHA = 4.0
SCALE = ALPHA / LORA_RANK

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

def lora_project(X, W, A, B):
    base = X @ W
    XA = X @ A
    update = XA @ B
    return base + SCALE * update, XA

def lora_project_backward(X, W, A, B, XA, dOut):
    dBase = dOut
    dUpdate = SCALE * dOut
    dXA = dUpdate @ B.T
    dB = XA.T @ dUpdate
    dX_fromA = dXA @ A.T
    dA = X.T @ dXA
    dX_fromW = dBase @ W.T
    return dX_fromW + dX_fromA, dA, dB

def attention_scores(Q, K):
    return (Q @ K.T) / np.sqrt(HEAD_DIM)

def attention_scores_backward(Q, K, dScores):
    draw = dScores / np.sqrt(HEAD_DIM)
    return draw @ K, draw.T @ Q

def attention_value_backward(probs, V, dOut):
    return dOut @ V.T, probs.T @ dOut

def mlp_backward(W1, W2, inner, dOut):
    d_inner_gelu = dOut @ W2.T
    d_inner = gelu_backward(inner, d_inner_gelu)
    return d_inner @ W1.T

def residual_backward(dOut):
    return dOut, dOut

# ------------------------------------------------------------
# Random weights.
# ------------------------------------------------------------

X = rng.standard_normal((TOKENS, EMBED_DIM)).astype(np.float32)

WQ = rng.standard_normal((EMBED_DIM, EMBED_DIM)).astype(np.float32)
WK = rng.standard_normal((EMBED_DIM, EMBED_DIM)).astype(np.float32)
WV = rng.standard_normal((EMBED_DIM, EMBED_DIM)).astype(np.float32)
WO = rng.standard_normal((EMBED_DIM, EMBED_DIM)).astype(np.float32)

AQ = rng.standard_normal((EMBED_DIM, LORA_RANK)).astype(np.float32)
BQ = rng.standard_normal((LORA_RANK, EMBED_DIM)).astype(np.float32)
AV = rng.standard_normal((EMBED_DIM, LORA_RANK)).astype(np.float32)
BV = rng.standard_normal((LORA_RANK, EMBED_DIM)).astype(np.float32)

W1 = rng.standard_normal((EMBED_DIM, MLP_DIM)).astype(np.float32)
W2 = rng.standard_normal((MLP_DIM, EMBED_DIM)).astype(np.float32)

# ------------------------------------------------------------
# Forward (caches everything the backward pass needs).
# ------------------------------------------------------------

def forward(X):
    layer1 = layer_norm(X)

    Q, XA_Q = lora_project(layer1, WQ, AQ, BQ)
    K = layer1 @ WK
    V, XA_V = lora_project(layer1, WV, AV, BV)

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

    cache = dict(layer1=layer1, Q=Q, K=K, V=V, XA_Q=XA_Q, XA_V=XA_V,
                 probs_cache=probs_cache, Qh_cache=Qh_cache,
                 Kh_cache=Kh_cache, Vh_cache=Vh_cache,
                 layer3=layer3, layer4=layer4, inner=inner)
    return out, cache

out, cache = forward(X)

# ------------------------------------------------------------
# Analytic backward, mirroring transformer_block_lora_backward_impl.
# ------------------------------------------------------------

def backward(dOut, cache):
    layer1, layer3, layer4, inner = (
        cache['layer1'], cache['layer3'], cache['layer4'], cache['inner'])

    dLayer3_a, dLayer5 = residual_backward(dOut)
    dLayer4 = mlp_backward(W1, W2, inner, dLayer5)
    dLayer3_b = layer_norm_backward(layer3, layer4, dLayer4)
    dLayer3 = dLayer3_a + dLayer3_b

    dX_a, dLayer2 = residual_backward(dLayer3)
    dConcatenated = dLayer2 @ WO.T

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

    dLayer1_q, dAQ, dBQ = lora_project_backward(layer1, WQ, AQ, BQ, cache['XA_Q'], dQ)
    dLayer1_k = dK @ WK.T
    dLayer1_v, dAV, dBV = lora_project_backward(layer1, WV, AV, BV, cache['XA_V'], dV)

    dLayer1 = dLayer1_q + dLayer1_k + dLayer1_v
    dX_b = layer_norm_backward(X, layer1, dLayer1)

    return dX_a + dX_b, dAQ, dBQ, dAV, dBV

dOut_seed = rng.standard_normal((TOKENS, EMBED_DIM)).astype(np.float32)
dX, dAQ, dBQ, dAV, dBV = backward(dOut_seed, cache)

# ------------------------------------------------------------
# Finite-difference sanity check on dX (independent of the
# analytic formulas above -- catches chain-rule mistakes).
# ------------------------------------------------------------

def directional_loss(Xp):
    o, _ = forward(Xp)
    return np.sum(o * dOut_seed)

h = 1e-3
fd_dX = np.zeros_like(X, dtype=np.float64)
for i in range(TOKENS):
    for j in range(EMBED_DIM):
        orig = X[i, j]
        X[i, j] = orig + h
        lp = directional_loss(X)
        X[i, j] = orig - h
        lm = directional_loss(X)
        X[i, j] = orig
        fd_dX[i, j] = (lp - lm) / (2 * h)

print("max |analytic dX - finite-diff dX| =", np.max(np.abs(dX - fd_dX)))

# ------------------------------------------------------------
# Flattened C arrays for the HLS testbench.
# ------------------------------------------------------------

def dump(name, arr):
    body = ", ".join(f"{v:.9f}f" for v in arr.flatten())
    print(f"float {name}[{arr.size}] = {{{body}}};")

print(f"\n// TOKENS={TOKENS} EMBED_DIM={EMBED_DIM} NUM_HEADS={NUM_HEADS} "
      f"LORA_RANK={LORA_RANK} MLP_DIM={MLP_DIM} ALPHA={ALPHA}")
for name, arr in [
    ("X", X), ("WQ", WQ), ("WK", WK), ("WV", WV), ("WO", WO),
    ("lora_AQ", AQ), ("lora_BQ", BQ), ("lora_AV", AV), ("lora_BV", BV),
    ("W1", W1), ("W2", W2), ("dOut_seed", dOut_seed),
    ("expected_out", out), ("expected_dX", dX),
    ("expected_dAQ", dAQ), ("expected_dBQ", dBQ),
    ("expected_dAV", dAV), ("expected_dBV", dBV),
]:
    dump(name, arr)
