"""Exports PASE (cisco-open/pase: DeWavLM + dual-stream Vocos vocoder) to ONNX.

  pase_encoder.onnx  wav (1, L) 16 kHz, L = 320k + 80 -> feat_a (1, T, 1024), feat_p (1, T, 1024)
                     (layers 1 and 24 of DeWavLM, each normalised over the whole clip)
  pase_vocoder.onnx  feat_p, feat_a (1, T, 1024) -> wav (1, 320 T)

The input padding/trimming to 320k + 80 samples and the final trim to the input
length are done in JavaScript, as PASE.forward does them in Python.

usage: PASE_SRC=<cisco-open/pase checkout> python export_pase.py <weights dir> <out dir> [test.wav]
"""
import math, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

torch.set_grad_enabled(False)
sys.path.insert(0, os.path.join(os.environ["PASE_SRC"], "pase"))
from models.pase import PASE  # noqa: E402


def global_norm(x, eps=1e-6):
    """F.layer_norm(x, x.shape): normalise over every element (the shape is dynamic)."""
    mean = x.mean()
    var = ((x - mean) ** 2).mean()
    return (x - mean) / torch.sqrt(var + eps)


class ConvISTFTSame(nn.Module):
    """vocos ISTFT(padding="same") with a Hann window: irfft, overlap-add via a
    transposed convolution, trim (win - hop) / 2 each side, divide by the window envelope."""

    def __init__(self, n_fft, hop):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        w = torch.hann_window(n_fft, periodic=True, dtype=torch.float64)
        f = n_fft // 2 + 1
        k = torch.arange(f, dtype=torch.float64)[:, None]
        n = torch.arange(n_fft, dtype=torch.float64)[None, :]
        ang = 2 * math.pi * k * n / n_fft
        c = torch.full((f, 1), 2.0, dtype=torch.float64)
        c[0] = 1
        c[-1] = 1
        basis = torch.cat([c * torch.cos(ang), -c * torch.sin(ang)], 0) / n_fft * w
        self.register_buffer("basis", basis[:, None, :].float())
        self.register_buffer("wsq", (w * w)[None, None, :].float())

    def forward(self, re, im):  # (b f t)
        y = F.conv_transpose1d(torch.cat([re, im], 1), self.basis, stride=self.hop)
        env = F.conv_transpose1d(torch.ones_like(re[:, :1, :]), self.wsq, stride=self.hop)
        p = (self.n_fft - self.hop) // 2
        return y[:, 0, p:-p] / env[:, 0, p:-p]


class Encoder(nn.Module):
    def __init__(self, pase):
        super().__init__()
        self.wavlm = pase.encoder.wavlm

    def forward(self, wav):
        res = self.wavlm.extract_features(wav, output_layer=24)[0]
        reps = res["layer_reps"]
        return global_norm(reps[1]), global_norm(reps[24])


class Vocoder(nn.Module):
    def __init__(self, pase):
        super().__init__()
        d = pase.decoder
        self.proj, self.backbone, self.out = d.proj, d.decoder, d.head.out
        self.istft = ConvISTFTSame(d.head.istft.n_fft, d.head.istft.hop_length)

    def forward(self, feat_p, feat_a):
        x = (feat_p + self.proj(feat_a)).transpose(1, 2)
        x = self.backbone(x)
        x = self.out(x.transpose(1, 2)).transpose(1, 2)
        mag, p = x.chunk(2, dim=1)
        mag = torch.clip(torch.exp(mag), max=1e2)
        return self.istft(mag * torch.cos(p), mag * torch.sin(p))


def pad_len(n):
    """PASE's WavLM_feat.pad: pad (or trim) to 320k + 80 samples."""
    return n if n % 320 == 80 else n // 320 * 320 + 80


def main():
    wdir, out = sys.argv[1:3]
    pase = PASE(dewavlm_ckpt_path=f"{wdir}/DeWavLM.pt", dewavlm_output_layer=[1, 24],
                vocoder_ckpt_path=f"{wdir}/Vocoder_Dual.pt").eval()
    enc, voc = Encoder(pase).eval(), Vocoder(pase).eval()

    g = torch.Generator().manual_seed(0)
    if len(sys.argv) > 3:
        import soundfile as sf
        test = torch.from_numpy(sf.read(sys.argv[3], dtype="float32")[0])[None]
    else:
        test = torch.randn(1, 16000 * 4, generator=g) * 0.1

    def run(x):
        n = x.shape[1]
        L = pad_len(n)
        xp = F.pad(x, (0, L - n)) if L >= n else x[:, :L]
        a, p = enc(xp)
        y = voc(p, a)
        return F.pad(y, (0, n - y.shape[1])) if y.shape[1] < n else y[:, :n]

    ref = pase(test)
    mine = run(test)
    err = (ref - mine).norm() / ref.norm()
    print(f"module rewrite vs PASE: rel err {err:.2e}")
    assert err < 1e-4

    x = torch.randn(1, pad_len(16000 * 3), generator=g) * 0.1
    torch.onnx.export(enc, (x,), f"{out}/pase_encoder.onnx", input_names=["wav"], output_names=["feat_a", "feat_p"],
                      dynamic_axes={"wav": {1: "L"}, "feat_a": {1: "T"}, "feat_p": {1: "T"}}, opset_version=17, dynamo=False)
    a, p = enc(x)
    torch.onnx.export(voc, (p, a), f"{out}/pase_vocoder.onnx", input_names=["feat_p", "feat_a"], output_names=["wav"],
                      dynamic_axes={"feat_p": {1: "T"}, "feat_a": {1: "T"}, "wav": {1: "N"}}, opset_version=17, dynamo=False)

    import onnxruntime as ort
    eo = ort.InferenceSession(f"{out}/pase_encoder.onnx")
    vo = ort.InferenceSession(f"{out}/pase_vocoder.onnx")
    n = test.shape[1]
    L = pad_len(n)
    xp = F.pad(test, (0, L - n)) if L >= n else test[:, :L]
    fa, fp = eo.run(None, {"wav": xp.numpy()})
    y = vo.run(None, {"feat_p": fp, "feat_a": fa})[0]
    y = np.pad(y, ((0, 0), (0, max(0, n - y.shape[1]))))[:, :n]
    e = np.linalg.norm(y - ref.numpy()) / np.linalg.norm(ref.numpy())
    print(f"ONNX vs PASE: rel err {e:.2e} ({n} samples)")
    assert e < 1e-3


if __name__ == "__main__":
    main()
