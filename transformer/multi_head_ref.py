import numpy as np

# ------------------------------------------------------------
# Test configuration
# ------------------------------------------------------------

TOKENS = 3
EMBED_DIM = 4
NUM_HEADS = 2

assert EMBED_DIM % NUM_HEADS == 0

HEAD_DIM = EMBED_DIM // NUM_HEADS
DIM_OUT = EMBED_DIM


# ------------------------------------------------------------
# Test inputs
# ------------------------------------------------------------

X = np.arange(1, TOKENS * EMBED_DIM + 1, dtype=np.float32)
X = X.reshape(TOKENS, EMBED_DIM)

WQ = np.arange(1, EMBED_DIM * EMBED_DIM + 1, dtype=np.float32)
WQ = WQ.reshape(EMBED_DIM, EMBED_DIM)

WK = np.arange(1, EMBED_DIM * EMBED_DIM + 1, dtype=np.float32)
WK = WK.reshape(EMBED_DIM, EMBED_DIM)

WV = np.arange(1, EMBED_DIM * EMBED_DIM + 1, dtype=np.float32)
WV = WV.reshape(EMBED_DIM, EMBED_DIM)

WO = np.arange(1, EMBED_DIM * DIM_OUT + 1, dtype=np.float32)
WO = WO.reshape(EMBED_DIM, DIM_OUT)


# ------------------------------------------------------------
# 1. Full Q, K, V projections
# ------------------------------------------------------------

Q = X @ WQ
K = X @ WK
V = X @ WV

print("Q:")
print(Q)

print("\nK:")
print(K)

print("\nV:")
print(V)


# ------------------------------------------------------------
# 2. Multi-head attention
# ------------------------------------------------------------

concatenated = np.zeros((TOKENS, EMBED_DIM), dtype=np.float32)

for h in range(NUM_HEADS):

    start = h * HEAD_DIM
    end = start + HEAD_DIM

    # Extract this head's columns
    Q_h = Q[:, start:end]
    K_h = K[:, start:end]
    V_h = V[:, start:end]

    print(f"\nHead {h}")
    print("Q_h:")
    print(Q_h)

    print("K_h:")
    print(K_h)

    print("V_h:")
    print(V_h)

    # --------------------------------------------------------
    # Attention scores
    # --------------------------------------------------------

    scores = (Q_h @ K_h.T) / np.sqrt(HEAD_DIM)

    print("Scaled scores:")
    print(scores)

    # --------------------------------------------------------
    # Row-wise stable softmax
    # --------------------------------------------------------

    shifted = scores - np.max(scores, axis=1, keepdims=True)

    exp_scores = np.exp(shifted)

    attention_weights = (
        exp_scores /
        np.sum(exp_scores, axis=1, keepdims=True)
    )

    print("Attention weights:")
    print(attention_weights)

    # --------------------------------------------------------
    # Attention weights × V
    # --------------------------------------------------------

    head_output = attention_weights @ V_h

    print("Head output:")
    print(head_output)

    # Put this head back into its section
    concatenated[:, start:end] = head_output


# ------------------------------------------------------------
# 3. Concatenated heads
# ------------------------------------------------------------

print("\nConcatenated heads:")
print(concatenated)


# ------------------------------------------------------------
# 4. Output projection
# ------------------------------------------------------------

expected = concatenated @ WO

print("\nExpected final output:")
print(expected)


# ------------------------------------------------------------
# Flattened output for C++ testbench
# ------------------------------------------------------------

print("\nC++ expected array:")

for value in expected.flatten():
    print(f"{value:.9f}f,", end=" ")

print()