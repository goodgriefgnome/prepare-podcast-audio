"""Exports Sidon v0.1 (sarulab-speech/sidon-v0.1) to ONNX.

The released checkpoints are frozen TorchScript traces, which the ONNX exporter
can't take directly (the predictor returns a dict). Their code is plain tensor
operations on constants, so it's translated into an ordinary nn.Module and
exported from that.

  sidon_predictor.onnx  input_features (1, T, 160) -> features (1, T, 1024)
  sidon_vocoder.onnx    features (1, T, 1024)      -> audio (1, T*960)  (48 kHz)

usage: python export_sidon.py <sidon dir with *_cpu.pt> <out dir>
"""
import re, sys, types
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

torch.set_grad_enabled(False)


def translate(script):
    code, consts = script.code_with_constants
    src = code
    src = re.sub(r"def forward\(self,\s*(\w+): Tensor\) -> [^:]+:", r"def forward(self, \1):", src)
    src = src.replace("CONSTANTS.", "self.")
    # helpers rather than regexes, since arguments can be nested calls
    src = src.replace("torch.to(", "_to(").replace("torch.size(", "_size(").replace("torch.view(", "_view(")
    src = re.sub(r'torch\.arange\((\w+), dtype=4, layout=None, device=torch\.device\("cpu"\), pin_memory=False\)', r"torch.arange(\1)", src)
    src = src.replace("torch.pad(", "F.pad(").replace("torch.silu(", "F.silu(").replace("torch.glu(", "F.glu(")
    src = re.sub(r'(\w+) = \{"last_hidden_state": (\w+), [^}]*\}', r"\1 = \2", src)
    leftover = set(re.findall(r"torch\.device|ops\.|prim|annotate|uninitialized", src))
    assert not leftover, leftover
    m = nn.Module()
    for k, v in consts.const_mapping.items():
        if isinstance(v, torch.Tensor):
            m.register_buffer(k, v.detach().clone().float() if v.is_floating_point() else v.clone())
        else:
            setattr(m, k, v)
    def _to(x, t):
        assert t == 6, t  # 6 = float32
        return x.to(torch.float32)

    ns = {"torch": torch, "F": F, "_to": _to,
          "_size": lambda x, i=None: x.shape if i is None else x.size(i),
          "_view": lambda x, s: torch.reshape(x, s)}
    exec(src, ns)
    m.forward = types.MethodType(ns["forward"], m)
    return m, src


def main():
    src_dir, out = sys.argv[1:3]
    fe = torch.jit.load(f"{src_dir}/feature_extractor_cpu.pt", map_location="cpu").eval()
    dec = torch.jit.load(f"{src_dir}/decoder_cpu.pt", map_location="cpu").eval()
    pm, _ = translate(fe)
    vm, _ = translate(dec)

    g = torch.Generator().manual_seed(0)
    for T in (37, 300):
        x = torch.randn(1, T, 160, generator=g)
        a = fe(x)["last_hidden_state"]
        b = pm(x)
        assert torch.allclose(a, b, atol=1e-4), (T, (a - b).abs().max())
        y = dec(a.transpose(1, 2))
        z = vm(a.transpose(1, 2))
        assert torch.allclose(y, z, atol=1e-4), (T, (y - z).abs().max())

    class Vocoder(nn.Module):
        def __init__(s):
            super().__init__()
            s.m = vm

        def forward(s, features):  # (1, T, 1024)
            return s.m(features.transpose(1, 2)).reshape(1, -1)

    x = torch.randn(1, 120, 160, generator=g)
    torch.onnx.export(pm, (x,), f"{out}/sidon_predictor.onnx", input_names=["input_features"], output_names=["features"],
                      dynamic_axes={"input_features": {1: "T"}, "features": {1: "T"}}, opset_version=17, dynamo=False)
    f = pm(x)
    torch.onnx.export(Vocoder(), (f,), f"{out}/sidon_vocoder.onnx", input_names=["features"], output_names=["audio"],
                      dynamic_axes={"features": {1: "T"}, "audio": {1: "N"}}, opset_version=17, dynamo=False)

    import onnxruntime as ort
    po = ort.InferenceSession(f"{out}/sidon_predictor.onnx")
    vo = ort.InferenceSession(f"{out}/sidon_vocoder.onnx")
    for T in (50, 411):
        x = torch.randn(1, T, 160, generator=g)
        a = fe(x)["last_hidden_state"]
        b = po.run(None, {"input_features": x.numpy()})[0]
        e1 = np.linalg.norm(a.numpy() - b) / np.linalg.norm(a.numpy())
        y = dec(a.transpose(1, 2)).reshape(1, -1).numpy()
        z = vo.run(None, {"features": a.numpy()})[0]
        e2 = np.linalg.norm(y - z) / np.linalg.norm(y)
        print(f"T={T}: predictor rel err {e1:.2e}, vocoder rel err {e2:.2e}, audio {z.shape}")
        assert e1 < 1e-3 and e2 < 1e-3


if __name__ == "__main__":
    main()
