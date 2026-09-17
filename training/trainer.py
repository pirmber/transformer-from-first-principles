"""
training/trainer.py
----------------------
Ties together the model, loss, optimizer, and dataset into a training loop.
Also records the statistics needed for the required visualizations:
loss curve, gradient norms per component, and parameter statistics.
"""

import numpy as np

from losses.cross_entropy import CrossEntropyLoss
from optim.adam import Adam
from tensor_utils import clip_grad_norm_, global_grad_norm, assert_finite


class Trainer:
    def __init__(self, model, dataset, config):
        self.model = model
        self.dataset = dataset
        self.config = config

        self.loss_fn = CrossEntropyLoss()
        self.layers = model.all_sub_layers()
        self.optimizer = Adam(self.layers, lr=config.learning_rate,
                               beta1=config.beta1, beta2=config.beta2,
                               eps=config.adam_eps)
        self.rng = np.random.default_rng(config.seed)

        self.history = {
            "step": [],
            "loss": [],
            "grad_norm": [],
        }
        # component_name -> list of per-step gradient norms
        self.grad_norm_by_component = {name: [] for name, _ in model.named_sub_layers()}

    def _record_gradient_norms(self):
        for name, layer in self.model.named_sub_layers():
            grads = list(layer.gradients().values())
            if len(grads) == 0:
                norm = 0.0
            else:
                norm = global_grad_norm(grads)
            self.grad_norm_by_component[name].append(norm)

    def train_step(self):
        x, y = self.dataset.get_batch(self.config.batch_size, self.rng)

        logits = self.model.forward(x)
        assert_finite(logits, "logits")
        loss = self.loss_fn.forward(logits, y)

        dlogits = self.loss_fn.backward()
        self.model.backward(dlogits)

        all_grads = []
        for layer in self.layers:
            all_grads.extend(layer.gradients().values())
        pre_clip_norm = clip_grad_norm_(all_grads, self.config.grad_clip_norm)

        self._record_gradient_norms()
        self.optimizer.step()

        return loss, pre_clip_norm

    def train(self, num_steps: int = None, log_every: int = 50, verbose: bool = True):
        num_steps = num_steps or self.config.num_steps
        for step in range(1, num_steps + 1):
            loss, grad_norm = self.train_step()

            self.history["step"].append(step)
            self.history["loss"].append(loss)
            self.history["grad_norm"].append(grad_norm)

            if verbose and (step % log_every == 0 or step == 1):
                print(f"step {step:5d} | loss {loss:.4f} | grad_norm {grad_norm:.4f}")

        return self.history
