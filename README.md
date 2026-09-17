# Transformer From Scratch

A tiny, decoder-only, GPT-style Transformer with **every forward and
backward operation implemented by hand in NumPy** — no PyTorch, TensorFlow,
JAX, autograd, or pretrained models anywhere in the codebase.

The goal of this project is not to produce a good language model. It's to
demonstrate, in runnable, verifiable code, exactly how gradients flow
through embeddings, LayerNorm, multi-head causal self-attention, residual
connections, a GELU feed-forward network, and cross-entropy loss — and to
prove those hand-derived gradients are correct using finite-difference
gradient checking.

## Motivation

Frameworks like PyTorch hide backpropagation behind `.backward()`. That's
great for building things quickly, but it also hides the actual math:
what gradient flows into a query projection when a token attends to five
different keys? What does the LayerNorm backward formula actually look
like once you account for the fact that both the mean *and* the variance
depend on every input element? Implementing these by hand — and checking
every one of them against numerical differentiation — is one of the most
direct ways to actually understand how a Transformer learns.

## Architecture

```text
                    INPUT TEXT
                        |
                        v
                    TOKENIZER (char-level)
                        |
                        v
                TOKEN ID SEQUENCE
                        |
                        v
        TOKEN EMBEDDINGS + POSITION EMBEDDINGS
                        |
                        v
        +------------------------------+
        |       TRANSFORMER BLOCK       |  x N (num_layers)
        |                                |
        |   x0 --> LayerNorm1 --> QKV    |
        |          --> Multi-Head        |
        |             Causal Attention   |
        |          --> Output Proj       |
        |   x1 = x0 + attn_out  (resid) |
        |                                |
        |   x1 --> LayerNorm2            |
        |       --> Linear -> GELU       |
        |       --> Linear (FFN)         |
        |   x2 = x1 + ffn_out   (resid) |
        +------------------------------+
                        |
                        v
                  Final LayerNorm
                        |
                        v
                  Final Linear (unembed)
                        |
                        v
                      LOGITS
                        |
                        v
                     SOFTMAX
                        |
                        v
                CROSS-ENTROPY LOSS
```

Each block uses **pre-LayerNorm** residual connections (LayerNorm is
applied before attention/FFN, not after), which is the standard modern
GPT-style arrangement and gives more stable gradients through deep stacks.

## Repository layout

```text
transformer-from-scratch/
├── config.py                 # TransformerConfig dataclass
├── tokenizer.py               # CharTokenizer
├── tensor_utils.py             # finite tensor checks, grad-norm/clipping
├── layers/
│   ├── linear.py               # Y = XW + b, manual backward
│   ├── embedding.py            # token + position lookup tables
│   ├── layernorm.py            # LayerNorm, manual analytical backward
│   ├── softmax.py               # stable softmax, VJP + Jacobian backward
│   └── gelu.py                  # tanh-approx GELU, manual backward
├── attention/
│   ├── causal_mask.py           # additive lower-triangular mask
│   └── multi_head_attention.py  # full manual attention backward chain
├── transformer/
│   ├── feed_forward.py           # Linear -> GELU -> Linear
│   ├── transformer_block.py      # pre-LN block with residuals
│   └── transformer.py            # full model (embeddings + N blocks + head)
├── losses/cross_entropy.py       # causal LM loss, probs - one_hot gradient
├── optim/adam.py                  # Adam optimizer, from scratch
├── training/
│   ├── dataset.py                  # sliding-window batching
│   └── trainer.py                   # training loop + stat recording
├── generation/generate.py           # autoregressive sampling (temp, top-k)
├── visualization/
│   ├── loss_plot.py
│   ├── attention_plot.py
│   └── gradient_plot.py
├── tests/                            # unit tests + gradient checking
└── train.py                          # end-to-end entrypoint
```

## Mathematics

### Chain rule & matrix derivatives

Every backward() method in this project is one application of the chain
rule: given the gradient of the loss with respect to a layer's *output*
(`dy`), compute the gradient with respect to its *inputs and parameters*.
For a matrix multiply `Y = XW`, the two matrix-calculus identities used
everywhere are:

```text
dL/dX = dL/dY @ W^T
dL/dW = X^T @ dL/dY
```

### Softmax derivative

For `s = softmax(x)`, the Jacobian is `ds_i/dx_j = s_i * (delta_ij - s_j)`.
Rather than forming this `(K, K)` matrix, the efficient vector-Jacobian
product is used everywhere in training:

```text
dx = s * (dy - sum(dy * s))
```

`layers/softmax.py` implements both forms and `tests/test_softmax.py`
verifies they agree.

### Attention

```text
S = QK^T / sqrt(d_k)        (masked with a large negative constant)
A = softmax(S)
O = A @ V
```

Backward (see `attention/multi_head_attention.py` for the full derivation
in code comments):

```text
dV = A^T @ dO
dA = dO @ V^T
dS = softmax_backward(dA, A)
dQ = dS @ K / sqrt(d_k)
dK = dS^T @ Q / sqrt(d_k)
```

Because the same input `X` produces `Q`, `K`, and `V` via three separate
projections, `dX = dX_from_Q + dX_from_K + dX_from_V`.

### LayerNorm

```text
mu = mean(x);  var = mean((x - mu)^2);  xhat = (x - mu) / sqrt(var + eps)
y = gamma * xhat + beta
```

The backward pass accounts for the fact that *both* `mu` and `var` are
functions of every element of `x`:

```text
dx = (1 / (N * std)) * ( N*dxhat - sum(dxhat) - xhat * sum(dxhat * xhat) )
```

### Residual connections

For `Y = X + F(X)`, the incoming gradient `dY` must flow down **both**
the identity path and into `F`'s backward: `dX = dY + dF/dX(dY)`. It's
easy to accidentally drop the identity term; `transformer_block.py` keeps
both paths explicit.

### GELU

This project uses the tanh approximation:
`GELU(x) = 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x^3)))`, with its exact
(not numerically approximated) derivative implemented in `layers/gelu.py`.

### Cross-entropy

The softmax + cross-entropy combination simplifies beautifully:

```text
dLoss/dLogits = probabilities - one_hot(target)
```

averaged over all `(batch, position)` pairs.

### Adam

Implemented from scratch in `optim/adam.py`: exponential moving averages
of the gradient (`m`) and squared gradient (`v`), bias-corrected, used to
scale the per-parameter learning rate.

## Forward pass

Token ids are embedded and summed with learned position embeddings, then
passed through `num_layers` Transformer blocks, a final LayerNorm, and a
linear layer that projects back up to vocabulary size to produce logits.

## Backward pass

`Transformer.backward()` walks the exact reverse of the forward pass:
final linear → final LayerNorm → each block in reverse → split into
token/position embedding gradients. Every intermediate value needed for
the backward pass is cached during `forward()` by each layer.

## Gradient verification

`tests/grad_check_utils.py` implements finite-difference gradient
checking: `grad ≈ (L(theta+eps) - L(theta-eps)) / (2*eps)`. Every major
component (Linear, LayerNorm, Softmax, GELU, Attention, TransformerBlock,
and the full model end-to-end) is checked this way in `tests/`. Run:

```bash
pip install -r requirements.txt
python -m pytest tests/ -q
python -m tests.test_gradients     # human-readable report
```

Sample output:

```text
[PASS] Linear dX: abs_error=5.460e-13  rel_error=1.652e-10
[PASS] LayerNorm dX: abs_error=3.953e-11  rel_error=2.197e-09
[PASS] Softmax dX: abs_error=6.190e-12  rel_error=1.993e-10
[PASS] GELU dX: abs_error=3.446e-11  rel_error=2.099e-09
[PASS] MultiHeadAttention dX: abs_error=2.270e-13  rel_error=1.923e-08
[PASS] TransformerBlock dX: abs_error=1.760e-10  rel_error=2.205e-08
```

Numerical gradient checking is used only for testing/debugging — it is
never used during actual training, which relies solely on the analytical
gradients computed by each layer's `backward()`.

## Training

```bash
python train.py
```

This will: run gradient checks, build a character tokenizer over a small
built-in corpus, construct a 2-layer / 4-head / 64-dim Transformer, train
it for 600 steps with Adam (gradient-clipped), save its parameters,
generate sample text, and write four plots to `outputs/`.

## Results

On the small built-in corpus, loss drops from ~3.4 (roughly `log(vocab
size)`, i.e. random guessing) to ~0.08 within 600 steps, and the model
learns to reproduce and recombine phrases from the training corpus.
See `outputs/loss.png`, `outputs/attention.png` (which clearly shows zero
attention weight on future tokens, confirming the causal mask),
`outputs/gradient_norms.png`, `outputs/parameter_stats.png`, and
`outputs/sample_generation.txt`.

## Limitations

This is an educational implementation, not a production or research
system. Notable simplifications:

- Pure NumPy on CPU: no GPU support, no batched kernel fusion, no
  KV-caching during generation (each generation step reruns the full
  forward pass over the current context window).
- Character-level tokenization only, no BPE/subword tokenizer.
- No dropout, weight tying, or other regularization/optimization tricks
  common in production Transformers.
- The corpus and model are intentionally tiny so the entire pipeline —
  including gradient checking — runs in seconds on a laptop CPU.

The focus throughout is mathematical transparency and verifiability, not
scale or performance.
