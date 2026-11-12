"""Exports VoiceFixer (mode 0: analysis + 44.1 kHz vocoder) to one ONNX graph.

  vf.onnx  wav (1, 1, 1323000) = 30 s at 44.1 kHz -> wav (1, 1, T')   (trim centre to 30 s)

The model's internal padding depends on the length, so the graph has a fixed
30 s input (VoiceFixer's own segment length); shorter audio is zero-padded.

usage: VF_SRC=... python export_vf.py <out_dir>
"""
import os, sys
# VF_SRC: the unpacked voicefixer package; its checkpoints must be in ~/.cache/voicefixer.
sys.path.insert(0, os.environ["VF_SRC"])
import numpy as np
import torch
from torch import nn
from voicefixer import VoiceFixer
from voicefixer.tools.pytorch_util import from_log

torch.set_grad_enabled(False)


class VFAll(nn.Module):
    def __init__(self, vf):
        super().__init__()
        self.m = vf._model

    def forward(self, wav):
        sp, _, _ = self.m.f_helper.wav_to_spectrogram_phase(wav)
        mel_orig = self.m.mel(sp.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        out = self.m(sp, mel_orig)
        mel = from_log(out["mel"])
        return self.m.vocoder(mel)


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    vf = VoiceFixer()
    vf._model.eval()
    mod = VFAll(vf).eval()
    x = torch.randn(1, 1, 44100 * 30) * 0.05
    path = f"{out}/vf.onnx"
    torch.onnx.export(mod, (x,), path, input_names=["wav"], output_names=["out"],
                      opset_version=17, dynamo=False)
    import onnxruntime as ort
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    y = s.run(None, {"wav": x.numpy()})[0]
    print("wrote", path, os.path.getsize(path) // 1_000_000, "MB; output", y.shape)


if __name__ == "__main__":
    main()
