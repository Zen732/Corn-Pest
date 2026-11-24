"""Version 2: the ready-made image network (cnn_features.NETWORK: EfficientNet-B2 since 25 Nov
2026, MobileNetV3-Large before; ImageNet weights) as a plain list of layers and numbers, so the
phone app (lib/offline/network.dart) and the backend can run it without PyTorch. The batch-norm
steps are folded into the convolutions (same math, fewer steps).

Run from the project root (after cnn_features.py has downloaded the weights once):
    ml/.venv/Scripts/python.exe ml/src/network_export.py
Writes assets/models/network.json (the layers) and network.bin (float32 numbers, little-endian),
copies them to backend/models/, then checks that NumpyNetwork (used by the backend) gives the same
numbers as PyTorch on a few photos. NumpyNetwork is in network.py (shared with the backend, which
has no PyTorch).

Layer list ("ops"), run in order on an H x W x C image (channels last):
    conv   k x k convolution, stride s, padding (k-1)/2, groups 1 or C (depthwise), bias, then
           activation relu / hswish / silu / none. Weights stored [k][k][in per group][out]
           (depthwise: [k][k][C]).
    se     squeeze-excitation: average over the image -> fc1 (act: relu or silu) -> fc2 (gate:
           hsigmoid or sigmoid) -> multiply each channel. fc1 stored [C][S], fc2 [S][C].
    save   remember the current image (start of a block with a shortcut)
    add    add the remembered image (end of that block)
    pool   average over the image -> the photo's numbers (960 for MobileNetV3, 1408 for B2)
"""
import json
import shutil
from pathlib import Path

import numpy as np

from network import NumpyNetwork

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets" / "models"
BACKEND = ROOT / "backend" / "models"


ACTS = {"ReLU": "relu", "Hardswish": "hswish", "SiLU": "silu"}
GATES = {"Hardsigmoid": "hsigmoid", "Sigmoid": "sigmoid"}


def blocks(model):
    """The network's steps in order: plain conv layers and blocks (MobileNetV3's InvertedResidual
    or EfficientNet's MBConv, which EfficientNet groups in stages)."""
    import torch
    for layer in model.features:
        if isinstance(layer, torch.nn.Sequential) and not hasattr(layer, "block") and \
                all(hasattr(b, "block") for b in layer):
            yield from layer  # an EfficientNet stage
        else:
            yield layer


def export():
    import torch
    from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation
    from cnn_features import LENGTH, NETWORK, SIZE, WEIGHTS, torch_model
    model = torch_model()
    ops, chunks, offset = [], [], 0

    def put(a):
        nonlocal offset
        a = np.ascontiguousarray(a, dtype="<f4")
        chunks.append(a.tobytes())
        start = offset
        offset += a.size
        return start

    def conv(cna):
        c, bn = cna[0], cna[1]
        act = "none"
        if len(cna) > 2:
            act = ACTS[type(cna[2]).__name__]
        s = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach().numpy()
        w = c.weight.detach().numpy() * s[:, None, None, None]  # (out, in/g, k, k)
        b = (bn.bias - bn.running_mean * bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach().numpy()
        k, g = c.kernel_size[0], c.groups
        if g == 1:
            stored = w.transpose(2, 3, 1, 0)  # k, k, in, out
        else:
            assert g == c.in_channels == c.out_channels
            stored = w[:, 0].transpose(1, 2, 0)  # k, k, C
        ops.append({"op": "conv", "in": c.in_channels, "out": c.out_channels, "k": k, "stride": c.stride[0],
                    "groups": g, "act": act, "w": put(stored), "b": put(b)})

    def se(m):
        c, sq = m.fc1.in_channels, m.fc1.out_channels
        ops.append({"op": "se", "c": c, "s": sq, "act": ACTS[type(m.activation).__name__],
                    "gate": GATES[type(m.scale_activation).__name__],
                    "w1": put(m.fc1.weight.detach().numpy()[:, :, 0, 0].T), "b1": put(m.fc1.bias.detach().numpy()),
                    "w2": put(m.fc2.weight.detach().numpy()[:, :, 0, 0].T), "b2": put(m.fc2.bias.detach().numpy())})

    for layer in blocks(model):
        if isinstance(layer, Conv2dNormActivation):
            conv(layer)
            continue
        # A block. EfficientNet's random skipping of blocks (stochastic depth) is for training only.
        if layer.use_res_connect:
            ops.append({"op": "save"})
        for part in layer.block:
            if isinstance(part, SqueezeExcitation):
                se(part)
            else:
                assert part[0].dilation == (1, 1)
                conv(part)
        if layer.use_res_connect:
            ops.append({"op": "add"})
    ops.append({"op": "pool"})
    spec = {"name": f"{NETWORK} {WEIGHTS.split('.')[1]} (torchvision), features + average pool",
            "input": [SIZE, SIZE, 3], "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
            "resize": "OpenCV INTER_AREA", "output": LENGTH, "floats": offset, "ops": ops}
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / "network.json").write_text(json.dumps(spec), encoding="utf-8")
    (ASSETS / "network.bin").write_bytes(b"".join(chunks))
    BACKEND.mkdir(parents=True, exist_ok=True)
    for f in ("network.json", "network.bin"):
        shutil.copy2(ASSETS / f, BACKEND / f)
    print(f"{len(ops)} steps, {offset} numbers ({offset * 4 / 1e6:.1f} MB) -> {ASSETS} and {BACKEND}")


def check():
    import pandas as pd
    from cnn_features import Network, load, split_file
    meta = pd.read_csv(ROOT / "ml" / "results" / "features_meta.csv", keep_default_na=False)
    rows = meta[(meta["split"] == "test") & (meta["aug"] == "")].sample(12, random_state=0)
    torch_net, np_net = Network(), NumpyNetwork(ASSETS)
    worst = 0.0
    for _, r in rows.iterrows():
        rgb = load(split_file(r))
        a, b = torch_net.vectors([rgb])[0], np_net.vector(rgb)
        worst = max(worst, float(np.abs(a - b).max() / (np.abs(a).max() + 1e-9)))
    print(f"numpy vs PyTorch on 12 test photos: largest difference {worst:.2e} (relative to the largest value)")
    assert worst < 1e-4, "the exported network does not match PyTorch"


def reference(n=24):
    """build/parity/network/: n test photos as raw RGB (decoded by OpenCV) with PyTorch's
    numbers, for test/network_parity_test.dart (the phone's network on the same pixels)."""
    import pandas as pd
    from cnn_features import Network, load, split_file
    out = ROOT / "build" / "parity" / "network"
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(ROOT / "ml" / "results" / "features_meta.csv", keep_default_na=False)
    rows = meta[(meta["split"] == "test") & (meta["aug"] == "")].sample(n, random_state=1)
    net, refs = Network(), []
    for i, (_, r) in enumerate(rows.iterrows()):
        rgb = load(split_file(r))
        (out / f"{i}.rgb").write_bytes(rgb.tobytes())
        refs.append({"file": f"{i}.rgb", "width": rgb.shape[1], "height": rgb.shape[0],
                     "vector": [float(v) for v in net.vectors([rgb])[0]]})
    (out / "refs.json").write_text(json.dumps(refs), encoding="utf-8")
    print(f"wrote {n} reference photos to {out}")


if __name__ == "__main__":
    export()
    check()
    reference()
