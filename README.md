Compact Convolutional Transformer in Vitis HLS (with LoRA Fine-Tuning)

An HLS C++ implementation of a Compact Convolutional Transformer (CCT) for FPGA, covering the convolutional tokenizer, the transformer encoder block, and a hand-derived backward pass for LoRA fine-tuning on-device. Every kernel is written as a templated C++ function so the same code runs in C simulation and synthesizes with Vitis HLS.

The backward pass is checked against an independent NumPy reference, and the forward output plus every gradient match within float32 rounding.

What's implemented
Stage	Module	Notes
Tokenizer	conv2d_impl, relu_impl, maxpool_impl, flatten_tokens_impl	3×3 stride-2 convs, 3×3 stride-2 max-pools
Attention	qkv_project_impl, attention_scores_impl, attention_softmax_impl, attention_value_impl, multi_head_impl	Scaled dot-product, per-head split/concat
Block	layer_norm_impl, mlp_impl (GEMM → GELU → GEMM), residual_add_impl	Pre-norm encoder block
LoRA forward	lora_project_impl, multi_head_lora_impl, transformer_block_lora	LoRA on the Q and V projections
Backward	cct_modules_backward.h	Full backprop and LoRA-only backprop
Tokenizer (32×32×3 image → 4 tokens × 256)
input 3×32×32
  → Conv 3→64   (stride 2) → ReLU → MaxPool → 64×8×8
  → Conv 64→256 (stride 2) → ReLU → MaxPool → 256×2×2
  → flatten → 4 tokens × 256 features
Transformer block
X ─► LayerNorm ─► Multi-Head Attention (LoRA on Q, V) ─► (+X) ─► LayerNorm ─► MLP ─► (+) ─► out

The synthesized top level uses 4 tokens, embedding dimension 256, MLP dimension 512, 4 heads, and LoRA rank 4. It exposes every weight buffer through its own m_axi bundle, with s_axilite control. The frozen projection $W$ gets a trainable low-rank update:
In transformer_block_lora_backward, only $A_Q, B_Q, A_V, B_V$ are trainable. The frozen weights ($W_Q, W_K, W_V, W_O, W_1, W_2$) only pass the gradient back toward $X$ and never produce weight gradients, which is where LoRA's memory and compute savings come from. $XA$ is recomputed in the backward pass instead of cached, since $r$ is small.

transformer_block_backward is the full-fine-tuning counterpart. It produces gradients for all six weight matrices, which gives a baseline to compare against. Each Python reference (*_ref.py) builds random weights with a fixed seed. It runs the forward pass and an analytic backward pass in NumPy float32, then prints the expected values that are pasted into the matching C++ testbench. Tolerances are relative (5e-3, or 2e-2 on the smallest LoRA gradients). The remaining differences come from summation order and from hls::erf vs. math.erf.



Next steps
Fixed-point (ap_fixed) datapath and resource/latency comparison against float32
Array partitioning and dataflow between stages (most loops are currently sequential)
Sequence pooling and classifier head to complete the CCT
Optimizer step (SGD/Adam) on the LoRA parameters in hardware
