#!/usr/bin/env python
"""Generate golden reference values from NumPy for NumJa accuracy tests.

Usage:
    .venv/Scripts/python.exe scripts/golden/generate_golden.py

Writes JSON files to bench/src/test/resources/golden/. Each file contains the
op name, a fixed input seed, expected outputs, and a relative tolerance.
Regenerate any time NumPy version changes — see GOLDEN.md.
"""
import json
import os

import numpy as np
from sklearn.linear_model import LinearRegression as SkLinearRegression

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "bench", "src", "test", "resources", "golden")
SEED = 42


def write(name, payload):
    path = os.path.join(OUT_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {path}")


def matmul_256():
    rng = np.random.default_rng(SEED)
    a = rng.random((256, 256))
    b = rng.random((256, 256))
    expected = (a @ b).flatten().tolist()
    write("matmul_256", {
        "op": "matmul",
        "seed": SEED,
        "shape": [256, 256],
        "inputs": {"a": a.flatten().tolist(), "b": b.flatten().tolist()},
        "expected": expected,
        "tolerance_rel": 1e-12,
    })


def sum_mean_1e6():
    rng = np.random.default_rng(SEED + 1)
    data = rng.random(1_000_000) * 2.0 - 1.0
    write("sum_mean_1e6", {
        "op": "sum_mean",
        "seed": SEED + 1,
        "inputs": {"data": data.tolist()},
        "expected": {
            "sum": float(np.sum(data)),
            "mean": float(np.mean(data)),
        },
        # sequential double summation drifts vs pairwise numpy sum: rel 1e-15
        "tolerance_rel": 1e-15,
    })


def softmax_1000_extreme():
    """Softmax over vector[1000] with values in [-700, 700] — boundary regime."""
    rng = np.random.default_rng(SEED + 2)
    x = rng.uniform(-700.0, 700.0, size=1000)
    shifted = x - np.max(x)          # stable max-shift trick (NumJa must do the same)
    e = np.exp(shifted)
    expected = (e / e.sum()).tolist()
    write("softmax_1000_extreme", {
        "op": "softmax",
        "seed": SEED + 2,
        "inputs": {"x": x.tolist()},
        "expected": expected,
        "tolerance_rel": 1e-12,
    })


def linear_regression_iris():
    """Fit on iris features -> predict petal_width from the other 3 columns."""
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
        "inputs": {"X": X.tolist(), "y": y.tolist()},
        "expected": {
            "coefficients": model.coef_.tolist(),
            "intercept": float(model.intercept_),
            "predictions": preds,
        },
        # closed-form lstsq vs sklearn's solver: rel 1e-9 is generous but tight enough
        "tolerance_rel": 1e-9,
    })


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    matmul_256()
    sum_mean_1e6()
    softmax_1000_extreme()
    linear_regression_iris()
    print("done.")
