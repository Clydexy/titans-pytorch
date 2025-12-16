# Titans PyTorch API Reference

This document summarizes the public APIs exposed by the package and shows how they fit together to build a simple network.

## Package entry points

Importing from the top-level package re-exports the core components:

```python
from titans_pytorch import (
    NeuralMemory, NeuralMemState, mem_state_detach,
    MemoryMLP, MemoryAttention, FactorizedMemoryMLP, MemorySwiGluMLP, GatedResidualMemoryMLP,
    MemoryAsContextTransformer,
)
```

## Neural memory module

### `NeuralMemory`

A differentiable associative memory block that learns to retrieve and store key/value updates during sequence processing.

**Key arguments**

- `dim` (int): Embedding dimension for queries, keys, and values.
- `chunk_size` (int | tuple[int, int], default `1`): Retrieval and storage chunk sizes; use a tuple for different retrieve/store sizes.
- `batch_size` (int | None): Optional fixed batch size for improved efficiency when using chunking.
- `dim_head` (int | None): Per-head dimension; defaults to `dim` when `heads == 1`.
- `heads` (int, default `1`): Number of attention-style heads used for retrieval and storage.
- `model` (nn.Module | None): Memory update network; defaults to `MemoryMLP(dim_head, depth=2, expansion_factor=4.)`.
- `store_memory_loss_fn` (Callable): Optional auxiliary loss on stored states; defaults to `F.mse_loss` via `default_loss_fn`.
- Additional flags such as `momentum`, `per_parameter_lr_modulation`, `qkv_receives_diff_views`, and `spectral_norm_surprises` enable the experimental behaviors explored in the paper.

**Calling the module**

```python
import torch
from titans_pytorch import NeuralMemory

mem = NeuralMemory(dim=384, chunk_size=64, heads=4)
seq = torch.randn(2, 1024, 384)
retrieved, mem_state = mem(seq)

# reuse cached memory when processing more tokens
next_tokens = torch.randn(2, 128, 384)
retrieved_tail, mem_state = mem(next_tokens, cache=mem_state)
```

The forward pass returns `(retrieved_sequence, new_state)`. Pass a previous `NeuralMemState` via `cache` to continue reading/writing memory across calls.

### `NeuralMemState` and `mem_state_detach`

- `NeuralMemState` is a named tuple carrying memory weights, cache, and bookkeeping. It is returned by `NeuralMemory` and can be fed back through `cache`.
- `mem_state_detach(state)` removes gradients from all tensors inside a state, which is useful when carrying memory across evaluation steps without backpropagating through the entire history.

## Memory model primitives

These classes define pluggable update rules for `NeuralMemory` and can also be used standalone.

- `MemoryMLP(dim, depth, expansion_factor=2.0)`: Standard multi-layer perceptron used in the original Titans paper.
- `GatedResidualMemoryMLP(dim, depth, expansion_factor=4.0)`: Adds gated residual connections and a final projection for more stable updates.
- `FactorizedMemoryMLP(dim, depth, k=32)`: Factorizes each weight matrix to reduce parameters while keeping capacity.
- `MemorySwiGluMLP(dim, depth=1, expansion_factor=4.0)`: SwiGLU-inspired feedforward with residual connections.
- `MemoryAttention(dim, scale=8.0, expansion_factor=2.0)`: Attention-style memory updater that mixes causal attention with a parallel feedforward branch.

**Example: swap in a gated residual memory model**

```python
from titans_pytorch import NeuralMemory, GatedResidualMemoryMLP

mem = NeuralMemory(
    dim=256,
    heads=2,
    model=GatedResidualMemoryMLP(dim=256, depth=3)
)
```

## Transformer wrapper

### `MemoryAsContextTransformer`

A sequence model that combines token embeddings, positional encodings, optional neural memory blocks, and MAC (Memory-As-Context) attention. It supports both training and autoregressive sampling.

**Key arguments**

- `num_tokens` (int): Vocabulary size.
- `dim` (int): Model dimension.
- `depth` (int): Number of transformer blocks.
- `segment_len` (int): Sliding/local attention window length.
- `num_persist_mem_tokens` (int): Number of learned persistent memory tokens to prepend.
- `num_longterm_mem_tokens` (int): Number of learned long-term memory tokens to interleave with the sequence.
- `neural_memory_model` / `neural_memory_kwargs`: Supply a custom `NeuralMemory` instance or configuration for per-layer neural memory modules.
- Other flags (e.g., `sliding_window_attn`, `use_flex_attn`, `neural_mem_gate_attn_output`) control attention style and how neural memory interacts with the main attention stream.

**Example: classifier-style forward pass**

```python
import torch
from titans_pytorch import MemoryAsContextTransformer

model = MemoryAsContextTransformer(
    num_tokens=32000,
    dim=512,
    depth=4,
    segment_len=128,
    num_persist_mem_tokens=4,
    num_longterm_mem_tokens=16,
)

input_ids = torch.randint(0, 32000, (1, 1025))
loss = model(input_ids, return_loss=True)
loss.backward()
```

**Example: autoregressive sampling**

```python
prompt = torch.randint(0, 32000, (1, 64))
samples = model.sample(prompt, seq_len=128, temperature=0.9)
```

## Attention experiments

Two experimental attention blocks are provided for memory research and can be dropped into custom architectures.

- `ImplicitMLPAttention(dim, mlp_hiddens, activation=nn.SiLU(), heads=8, talking_heads=True, prenorm=True, keys_rmsnorm=True)`: Implements the implicit multi-layer perceptron attention proposed for Titans/TTT. Accepts a sequence of hidden sizes via `mlp_hiddens` and returns outputs matching the input dimensionality.
- `NestedAttention(dim, dim_head=64, heads=8, prenorm=True, keys_rmsnorm=True)`: Builds nested key/value chains before a second round of attention, allowing hierarchical aggregation.

**Example: drop-in usage**

```python
import torch
from titans_pytorch.implicit_mlp_attention import ImplicitMLPAttention

block = ImplicitMLPAttention(dim=512, mlp_hiddens=(64, 128, 128, 64))
seq = torch.randn(1, 256, 512)
outputs, cache = block(seq, return_kv_cache=True)
```

Both modules accept an optional `(keys, values)` cache so you can feed additional tokens incrementally during inference.

## Putting it together: simple network sketch

The components can be combined to build a compact memory-augmented model:

```python
import torch
from torch import nn
from titans_pytorch import NeuralMemory, MemoryMLP
from titans_pytorch.implicit_mlp_attention import ImplicitMLPAttention

class TinyTitans(nn.Module):
    def __init__(self):
        super().__init__()
        self.mem = NeuralMemory(dim=128, heads=2, chunk_size=32, model=MemoryMLP(128, depth=2))
        self.attn = ImplicitMLPAttention(dim=128, mlp_hiddens=(64, 128, 64))
        self.readout = nn.Linear(128, 10)

    def forward(self, tokens, mem_state=None):
        retrieved, mem_state = self.mem(tokens, cache=mem_state)
        attended = self.attn(retrieved)
        logits = self.readout(attended.mean(dim=1))
        return logits, mem_state

tokens = torch.randn(8, 100, 128)
model = TinyTitans()
logits, mem_state = model(tokens)
```

This sketch shows how `NeuralMemory` can maintain cross-token state, while `ImplicitMLPAttention` provides causal attention before a final prediction head.
