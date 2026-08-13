import numpy as np

TOKENS = 3
DIM = 4
DIM_OUT = 4

X = np.arange(1, 13, dtype=np.float32).reshape(TOKENS, DIM)

WQ = np.arange(1, 17, dtype=np.float32).reshape(DIM, DIM)
WK = np.arange(1, 17, dtype=np.float32).reshape(DIM, DIM)
WV = np.arange(1, 17, dtype=np.float32).reshape(DIM, DIM)
WO = np.arange(1, 17, dtype=np.float32).reshape(DIM, DIM_OUT)

Q = X @ WQ
K = X @ WK
V = X @ WV
scores = (Q @ K.T) / np.sqrt(DIM)
scores_shifted = scores - np.max(scores, axis=1, keepdims=True)

exp_scores = np.exp(scores_shifted)

attention_weights = (
    exp_scores /
    np.sum(exp_scores, axis=1, keepdims=True)
)
attention_output = attention_weights @ V
expected = attention_output @ WO

print("Q:")
print(Q)

print("\nK:")
print(K)

print("\nV:")
print(V)

print("\nScores:")
print(scores)

print("\nAttention weights:")
print(attention_weights)

print("\nAttention x V:")
print(attention_output)

print("\nExpected final output:")
print(expected)