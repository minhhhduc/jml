#!/usr/bin/env python
"""Generate golden reference values from NumPy for NumJa accuracy tests.

Usage:
    .venv/Scripts/python.exe scripts/golden/generate_golden.py

Writes JSON files to bench/src/test/resources/golden/. Each file contains the
op name, an input SEED, expected outputs, and a relative tolerance.

Inputs are NOT stored in the JSON: they are regenerated deterministically on
the Java side from the seed via java.util.Random. This script reimplements
Java's Random LCG exactly (48-bit, documented in the JDK Javadoc) so both
languages produce bit-identical inputs. Keeps golden files tiny instead of
tens of MB of raw doubles in git.

Regenerate any time NumPy version changes — see bench/GOLDEN.md.
"""
import json
import os

import numpy as np
from sklearn.linear_model import LinearRegression as SkLinearRegression

OUT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "bench", "src", "test", "resources", "golden"))

# --- java.util.Random compatible LCG (see JDK Javadoc for java.util.Random) ---
MASK = (1 << 48) - 1
MULT = 0x5DEECE66D
ADD = 0xB


class JavaRandom:
    def __init__(self, seed):
        self.seed = (seed ^ MULT) & MASK

    def next(self, bits):
        self.seed = (self.seed * MULT + ADD) & MASK
        return self.seed >> (48 - bits)

    def next_double(self):
        a = self.next(26)
        b = self.next(27)
        return ((a << 27) | b) * (2.0 ** -53)


def rand_doubles(rng, n):
    return np.array([rng.next_double() for _ in range(n)])


def write(name, payload):
    path = os.path.join(OUT_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"wrote {os.path.getsize(path):>9,} bytes  {path}")


def matmul_256():
    """Inputs: two [256][256] row-major matrices from java Random(seed)."""
    seed = 42
    rng = JavaRandom(seed)
    n = 256
    a = rand_doubles(rng, n * n).reshape(n, n)
    b = rand_doubles(rng, n * n).reshape(n, n)
    expected = (a @ b).flatten().tolist()
    write("matmul_256", {
        "op": "matmul",
        "seed": seed,
        "input_gen": "row-major [256][256] matrix 'a', then 'b', each via java.util.Random.nextDouble()",
        "shape": [n, n],
        "expected": expected,
        # EJML blocked matmul vs NumPy BLAS: different accumulation order -> rel 1e-12
        "tolerance_rel": 1e-12,
    })


def sum_mean():
    """Inputs: double[n] uniform in [-1, 1] via (rng.nextDouble()*2-1)."""
    seed = 43
    n = 1_000_000
    rng = JavaRandom(seed)
    data = rand_doubles(rng, n) * 2.0 - 1.0
    write("sum_mean", {
        "op": "sum_mean",
        "seed": seed,
        "input_gen": "double[1000000]: java.util.Random.nextDouble()*2-1 per element",
        "expected": {
            "sum": float(np.sum(data)),
            "mean": float(np.mean(data)),
        },
        # Sequential double summation (NDArray.sum) vs pairwise NumPy sum drifts
        # ~1e-14 at n=1e6 — tolerance must sit above that unavoidable drift.
        "tolerance_rel": 1e-13,
    })


def softmax_extreme():
    """Inputs: double[1000] uniform in [-700, 700] via (rng.nextDouble()*1400-700)."""
    seed = 44
    n = 1000
    rng = JavaRandom(seed)
    x = rand_doubles(rng, n) * 1400.0 - 700.0
    shifted = x - np.max(x)  # stable max-shift (both sides must do this)
    e = np.exp(shifted)
    expected = (e / e.sum()).tolist()
    write("softmax_extreme", {
        "op": "softmax",
        "seed": seed,
        "input_gen": "double[1000]: java.util.Random.nextDouble()*1400-700 per element",
        "expected": expected,
        # max-shift exp accumulation: rel 1e-12
        "tolerance_rel": 1e-12,
    })


def linear_regression_iris():
    """Fit iris features -> predict petal_width from the other 3 columns."""
    raw = np.genfromtxt(
        os.path.join(os.path.dirname(__file__), "..", "..", "dist", "datasets", "iris.csv"),
        delimiter=",", skip_header=1, usecols=(0, 1, 2, 3),
    )
    X = raw[:, :3]
    y = raw[:, 3]
    model = SkLinearRegression().fit(X, y)
    preds = model.predict(X).tolist()
    write("linear_regression_iris", {
        "op": "linear_regression",
        "seed": None,
        "input_source": "dist/datasets/iris.csv cols 0-2 = X, col 3 = y",
        "expected": {
            "coefficients": model.coef_.tolist(),
            "intercept": float(model.intercept_),
            "predictions": preds,
        },
        # Normal equation + Gaussian elimination vs sklearn lstsq solver: rel 1e-9
        "tolerance_rel": 1e-9,
    })


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    matmul_256()
    sum_mean()
    softmax_extreme()
    linear_regression_iris()
    print("done.")
