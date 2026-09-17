"""
generation/generate.py
-------------------------
Autoregressive sampling from the trained model. Pure NumPy -- no external
generation/sampling library.

At every step:
  1. Take the current sequence (truncated to context_length if needed).
  2. Run the forward pass.
  3. Take the logits for the final position.
  4. Optionally apply temperature scaling and top-k filtering.
  5. Convert to probabilities with softmax and sample (or argmax if
     temperature == 0).
  6. Append the sampled token and repeat.
"""

import numpy as np

from layers.softmax import softmax_forward


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 200,
             temperature: float = 1.0, top_k: int = None,
             rng: np.random.Generator = None) -> str:
    rng = rng or np.random.default_rng()
    context_length = model.config.context_length

    ids = tokenizer.encode(prompt)
    if len(ids) == 0:
        # Need at least one token to seed generation.
        ids = [0]

    for _ in range(max_new_tokens):
        window = ids[-context_length:]
        x = np.array([window], dtype=np.int64)   # (1, T)

        logits = model.forward(x)                 # (1, T, V)
        last_logits = logits[0, -1, :].copy()      # (V,)

        if temperature == 0:
            next_id = int(np.argmax(last_logits))
        else:
            last_logits = last_logits / max(temperature, 1e-8)

            if top_k is not None and top_k < last_logits.shape[0]:
                # Keep only the top_k logits, mask out the rest.
                kth_value = np.partition(last_logits, -top_k)[-top_k]
                last_logits = np.where(last_logits < kth_value, -1e10, last_logits)

            probs = softmax_forward(last_logits, axis=-1)
            next_id = int(rng.choice(len(probs), p=probs))

        ids.append(next_id)

    return tokenizer.decode(ids)
