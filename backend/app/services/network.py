"""Version 2: runs the exported image network (network.json + network.bin, made by
network_export.py; EfficientNet-B2 since 25 Nov 2026) with numpy only, so the backend needs no
PyTorch. Shared with the backend like features.py (backend/app/services/network.py must be an
exact copy). The phone app runs the same steps in lib/offline/network.dart. Gives PyTorch's
numbers to about 1e-6.

    from network import NumpyNetwork
    net = NumpyNetwork("backend/models")
    v = net.vector(rgb)          # rgb: uint8 array (H, W, 3) -> float32 array (1408 numbers for B2)
"""
import json
from pathlib import Path

import cv2
import numpy as np


def _sigmoid(x):
    with np.errstate(over="ignore"):  # exp(-x) of a very negative x is inf: the answer is then 0
        return np.float32(1) / (np.float32(1) + np.exp(-x))


def _act(x, act):
    if act == "relu":
        return np.maximum(x, 0)
    if act == "hswish":
        return x * np.clip(x + 3, 0, 6) / 6
    if act == "silu":
        return x * _sigmoid(x)
    return x


class NumpyNetwork:
    """Runs network.json + network.bin with numpy (the backend's copy of the phone's network)."""

    def __init__(self, folder):
        folder = Path(folder)
        self.spec = json.loads((folder / "network.json").read_text(encoding="utf-8"))
        self.w = np.fromfile(folder / "network.bin", dtype="<f4")
        self.mean = np.array(self.spec["mean"], np.float32)
        self.std = np.array(self.spec["std"], np.float32)

    def _get(self, start, *shape):
        return self.w[start: start + int(np.prod(shape))].reshape(shape)

    def prepare(self, rgb):
        size = self.spec["input"][0]
        x = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        return (x - self.mean) / self.std  # H, W, C

    def _conv(self, x, op):
        k, s, p = op["k"], op["stride"], (op["k"] - 1) // 2
        h, w, c = x.shape
        oh, ow = (h + 2 * p - k) // s + 1, (w + 2 * p - k) // s + 1
        b = self._get(op["b"], op["out"])
        if k == 1 and op["groups"] == 1:
            out = x[::s, ::s].reshape(-1, c) @ self._get(op["w"], c, op["out"])
            return _act(out.reshape(oh, ow, -1) + b, op["act"])
        xp = np.pad(x, ((p, p), (p, p), (0, 0)))
        if op["groups"] == 1:
            wt = self._get(op["w"], k, k, c, op["out"])
            out = np.zeros((oh, ow, op["out"]), np.float32)
            for i in range(k):
                for j in range(k):
                    out += xp[i: i + s * oh: s, j: j + s * ow: s] @ wt[i, j]
        else:
            wt = self._get(op["w"], k, k, c)
            out = np.zeros((oh, ow, c), np.float32)
            for i in range(k):
                for j in range(k):
                    out += xp[i: i + s * oh: s, j: j + s * ow: s] * wt[i, j]
        return _act(out + b, op["act"])

    def _se(self, x, op):
        c, sq = op["c"], op["s"]
        v = x.mean(axis=(0, 1))
        v = _act(v @ self._get(op["w1"], c, sq) + self._get(op["b1"], sq), op.get("act", "relu"))
        v = v @ self._get(op["w2"], sq, c) + self._get(op["b2"], c)
        gate = _sigmoid(v) if op.get("gate", "hsigmoid") == "sigmoid" else np.clip(v + 3, 0, 6) / 6
        return x * gate

    def vector(self, rgb):
        """uint8 RGB photo -> the network's numbers (float32)."""
        x, saved = self.prepare(rgb), None
        for op in self.spec["ops"]:
            kind = op["op"]
            if kind == "conv":
                x = self._conv(x, op)
            elif kind == "se":
                x = self._se(x, op)
            elif kind == "save":
                saved = x
            elif kind == "add":
                x = x + saved
            elif kind == "pool":
                x = x.mean(axis=(0, 1))
        return x.astype(np.float32)
