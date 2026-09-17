"""
tests/grad_check_utils.py
----------------------------
Shared numerical (finite-difference) gradient checking utility, used by
every test file and by tests/test_gradients.py. This is NOT used during
normal training -- it exists purely to verify that the hand-derived
analytical gradients implemented throughout the project are correct.

    gradient ~= [L(theta + eps) - L(theta - eps)] / (2 * eps)
"""

import numpy as np


def numerical_gradient(f, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    f: function taking x (ndarray) and returning a scalar loss.
    x: the array to differentiate with respect to.
    Returns an array of the same shape as x containing the numerical
    gradient dL/dx via central differences.
    """
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]

        x[idx] = orig + eps
        loss_plus = f(x)

        x[idx] = orig - eps
        loss_minus = f(x)

        x[idx] = orig
        grad[idx] = (loss_plus - loss_minus) / (2 * eps)

        it.iternext()
    return grad


def relative_error(analytical: np.ndarray, numerical: np.ndarray) -> float:
    num = np.abs(analytical - numerical)
    denom = np.maximum(1e-8, np.abs(analytical) + np.abs(numerical))
    return float(np.max(num / denom))


def report(name: str, analytical: np.ndarray, numerical: np.ndarray, tol: float = 1e-3) -> bool:
    abs_err = float(np.max(np.abs(analytical - numerical)))
    rel_err = relative_error(analytical, numerical)
    passed = rel_err < tol
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}: abs_error={abs_err:.3e}  rel_error={rel_err:.3e}")
    return passed
