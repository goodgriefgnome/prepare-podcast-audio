"""Exports Resemble Enhance (enhancer) to ONNX as three graphs.

  re_front.onnx    wav (1, T)  -> mel_orig (1,128,F), mel_den (1,128,F), z0 (1,64,F)
  re_velocity.onnx psi (1,64,F), cond (1,128,F), t (1,) -> v (1,64,F)
  re_decode.onnx   psi1 (1,64,F) -> h (1,160,F)
  re_vocoder.onnx  h (1,160,F), noise (1,128,F+10) -> wav (1, F*420)
                   (memory-hungry: run on short overlapping windows)

The ODE loop, noise and chunking run in JavaScript. STFTs are done with
convolutions so the graphs only use common operators.

usage: RE_SRC=... RE_RUN=... python export_re.py <out_dir>
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from load_re import load

torch.set_grad_enabled(False)


def hann(n):
    return torch.hann_window(n, periodic=True, dtype=torch.float64)


class ConvSTFT(nn.Module):
    """torch.stft(center=True) as a strided convolution; returns real, imag (b f t)."""

    def __init__(self, n_fft, hop, win, pad_mode):
        super().__init__()
        self.n_fft, self.hop, self.pad_mode = n_fft, hop, pad_mode
        w = torch.zeros(n_fft, dtype=torch.float64)
        off = (n_fft - win) // 2
        w[off:off + win] = hann(win)
        k = torch.arange(n_fft // 2 + 1, dtype=torch.float64)[:, None]
        n = torch.arange(n_fft, dtype=torch.float64)[None, :]
        ang = 2 * math.pi * k * n / n_fft
        basis = torch.cat([torch.cos(ang) * w, -torch.sin(ang) * w], 0)  # (2f, n_fft)
        self.register_buffer("basis", basis[:, None, :].float())

    def forward(self, x):  # (b, T)
        p = self.n_fft // 2
        x = F.pad(x[:, None, :], (p, p), mode=self.pad_mode)
        s = F.conv1d(x, self.basis, stride=self.hop)
        f = self.n_fft // 2 + 1
        return s[:, :f], s[:, f:]


class ConvISTFT(nn.Module):
    """torch.istft(center=True, no length) with a Hann window, via transposed convolution."""

    def __init__(self, n_fft, hop):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        w = hann(n_fft)
        f = n_fft // 2 + 1
        k = torch.arange(f, dtype=torch.float64)[:, None]
        n = torch.arange(n_fft, dtype=torch.float64)[None, :]
        ang = 2 * math.pi * k * n / n_fft
        c = torch.full((f, 1), 2.0, dtype=torch.float64)
        c[0] = 1
        if n_fft % 2 == 0:
            c[-1] = 1
        basis = torch.cat([c * torch.cos(ang), -c * torch.sin(ang)], 0) / n_fft * w  # (2f, n_fft)
        self.register_buffer("basis", basis[:, None, :].float())
        self.register_buffer("wsq", (w * w)[None, None, :].float())

    def forward(self, re, im):  # (b f t)
        s = torch.cat([re, im], 1)
        y = F.conv_transpose1d(s, self.basis, stride=self.hop)
        ones = torch.ones_like(re[:, :1, :])
        env = F.conv_transpose1d(ones, self.wsq, stride=self.hop)
        p = self.n_fft // 2
        y = y[:, 0, p:-p]
        env = env[:, 0, p:-p]
        return y / torch.clamp(env, min=1e-11)


def patch_denoiser(d):
    cfg = d.stft_cfg
    stft = ConvSTFT(cfg["n_fft"], cfg["hop_length"], cfg["win_length"], "reflect")
    istft = ConvISTFT(cfg["n_fft"], cfg["hop_length"])
    d.add_module("conv_stft", stft)
    d.add_module("conv_istft", istft)

    def _stft(x):
        re, im = stft(x)
        re, im = re[..., :-1], im[..., :-1]
        mag = torch.sqrt(re * re + im * im)
        # angle() of (0,0) is 0, so cos 1, sin 0
        safe = mag > 0
        cos = torch.where(safe, re / torch.where(safe, mag, torch.ones_like(mag)), torch.ones_like(mag))
        sin = torch.where(safe, im / torch.where(safe, mag, torch.ones_like(mag)), torch.zeros_like(mag))
        return mag, cos, sin

    def _istft(mag, cos, sin):
        re, im = mag * cos, mag * sin
        re = torch.cat([re, re[..., -1:]], -1)
        im = torch.cat([im, im[..., -1:]], -1)
        return istft(re, im)

    d._stft, d._istft = _stft, _istft


class ConvMel(nn.Module):
    """Resemble's MelSpectrogram (preemphasis, |STFT|, slaney mel, dB, normalize)."""

    def __init__(self, mel_fn):
        super().__init__()
        hp = mel_fn.hp
        ms = mel_fn.melspec
        self.stft = ConvSTFT(hp.n_fft, hp.hop_size, hp.win_size, "constant")
        self.register_buffer("fb", ms.mel_scale.fb.T.contiguous())  # (mels, f)
        self.pre = hp.preemphasis
        self.min_mag = hp.stft_magnitude_min
        self.min_db = mel_fn.min_level_db

    def forward(self, wav):
        if self.pre > 0:
            wav = torch.cat([wav[..., :1], wav[..., 1:] - self.pre * wav[..., :-1]], -1)
        re, im = self.stft(wav)
        mag = torch.sqrt(re * re + im * im)
        mel = torch.matmul(self.fb, mag)
        db = torch.log10(torch.clamp(mel, min=self.min_mag)) * 20
        return (db - self.min_db) / (-self.min_db + 15)


def lvc_matmul(self, x, kernel, bias, dilation=1, hop_size=256):
    """location_variable_convolution without unfold/einsum (dilation 1)."""
    assert dilation == 1
    b, i, L = x.shape
    _, _, o, ks, K = kernel.shape
    p = (ks - 1) // 2
    xp = F.pad(x, (p, p))
    taps = torch.stack([xp[:, :, k:k + L] for k in range(ks)], 2)  # (b i k L)
    taps = taps.reshape(b, i, ks, K, hop_size)  # (b i k l h)
    taps = taps.permute(0, 3, 4, 1, 2).reshape(b, K, hop_size, i * ks)  # (b l h ik)
    ker = kernel.permute(0, 4, 1, 3, 2).reshape(b, K, i * ks, o)  # (b l ik o)
    out = torch.matmul(taps, ker)  # (b l h o)
    out = out + bias.permute(0, 2, 1)[:, :, None, :]  # bias (b o l) -> (b l 1 o)
    return out.permute(0, 3, 1, 2).reshape(b, o, K * hop_size)


def explicit_same_padding(model):
    """padding="same" exports as auto_pad, which ONNX Runtime rejects with dilation."""
    n = 0
    for mod in model.modules():
        if isinstance(mod, nn.Conv1d) and mod.padding == "same":
            k, d = mod.kernel_size[0], mod.dilation[0]
            assert (d * (k - 1)) % 2 == 0
            mod.padding = (d * (k - 1) // 2,)
            mod._reversed_padding_repeated_twice = [d * (k - 1) // 2] * 2
            n += 1
    print("explicit padding on", n, "convs")


def patch_alias_free():
    """The anti-aliasing filters build their grouped-conv kernel from the traced channel
    count, which the exporter can't size; channel counts are fixed, so make them constants."""
    from resemble_enhance.enhancer.univnet.alias_free_torch import resample, filter as flt

    def up(self, x):
        C = int(x.shape[1])
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * F.conv_transpose1d(x, self.filter.expand(C, -1, -1).contiguous(), stride=self.stride, groups=C)
        return x[..., self.pad_left:-self.pad_right]

    def low(self, x):
        C = int(x.shape[1])
        if self.padding:
            x = F.pad(x, (self.pad_left, self.pad_right), mode=self.padding_mode)
        return F.conv1d(x, self.filter.expand(C, -1, -1).contiguous(), stride=self.stride, groups=C)

    resample.UpSample1d.forward = up
    flt.LowPassFilter1d.forward = low


class Front(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        self.mel = ConvMel(m.mel_fn)

    def norm(self, x):
        return (x - self.m.normalizer.running_mean) / self.m.normalizer.running_std

    def forward(self, wav):
        x = wav / (wav.abs().max(dim=-1, keepdim=True).values + 1e-7)
        mel_orig = self.norm(self.mel(x)[..., :-1])
        den = self.m.denoiser(x)
        mel_den = self.norm(self.mel(den)[..., :-1])
        z0 = self.m.lcfm.ae.encoder(mel_orig) * self.m.lcfm.z_scale
        return mel_orig, mel_den, z0


class Velocity(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.cfm = m.lcfm.cfm

    def forward(self, psi, cond, t):
        g = self.cfm.emb(t.clamp(0, 1))
        return self.cfm.net(psi, l=cond, g=g)


class Decode(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, psi1):
        return self.m.lcfm.ae.decode(psi1 / self.m.lcfm.z_scale)


class Vocoder(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, h, noise):
        v = self.m.vocoder
        x = F.pad(h, (0, 10))
        z = v.conv_pre(noise)
        for block in v.blocks:
            z = block(z, x)
        z = v.conv_post(z)
        z = z[..., : -v.scale_factor * 10]
        return z.squeeze(1)


def export(module, args, names_in, names_out, dyn, path):
    torch.onnx.export(module, args, path, input_names=names_in, output_names=names_out,
                      dynamic_axes=dyn, opset_version=17, dynamo=False, do_constant_folding=True)
    print("wrote", path, os.path.getsize(path) // 1_000_000, "MB (+ data)" if os.path.exists(path + ".data") else "MB")


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    m = load()
    explicit_same_padding(m)
    # Check the rewritten pieces against the originals before exporting.
    torch.manual_seed(0)
    wav = torch.randn(1, 44100 * 3) * 0.1
    d = (m.mel_fn(wav) - ConvMel(m.mel_fn)(wav)).abs()
    print(f"mel: max diff {d.max().item():.2e}, mean {d.mean().item():.2e} (normalised units)")
    assert d.mean() < 1e-3
    ref_den = m.denoiser(wav)
    patch_denoiser(m.denoiser)
    d = (ref_den - m.denoiser(wav)).abs().max().item()
    print(f"denoiser: max diff {d:.2e}")
    assert d < 1e-6
    from resemble_enhance.enhancer.univnet import lvcnet
    x, k, b = torch.randn(1, 96, 7 * 20), torch.randn(1, 96, 192, 3, 20), torch.randn(1, 192, 20)
    d = (lvcnet.LVCBlock.location_variable_convolution(None, x, k, b, hop_size=7) - lvc_matmul(None, x, k, b, hop_size=7)).abs().max().item()
    print(f"location-variable convolution: max diff {d:.2e}")
    assert d < 1e-4
    lvcnet.LVCBlock.location_variable_convolution = lvc_matmul
    patch_alias_free()

    T = 44100 * 4
    Fr = T // 420
    export(Front(m).eval(), (torch.randn(1, T),), ["wav"], ["mel_orig", "mel_den", "z0"],
           {"wav": {1: "T"}, "mel_orig": {2: "F"}, "mel_den": {2: "F"}, "z0": {2: "F"}}, f"{out}/re_front.onnx")
    export(Velocity(m).eval(), (torch.randn(1, 64, Fr), torch.randn(1, 128, Fr), torch.tensor([0.3])),
           ["psi", "cond", "t"], ["v"], {"psi": {2: "F"}, "cond": {2: "F"}, "v": {2: "F"}}, f"{out}/re_velocity.onnx")
    export(Decode(m).eval(), (torch.randn(1, 64, Fr),), ["psi1"], ["h"],
           {"psi1": {2: "F"}, "h": {2: "F"}}, f"{out}/re_decode.onnx")
    export(Vocoder(m).eval(), (torch.randn(1, 160, 100), torch.randn(1, 128, 110)), ["h", "noise"], ["wav"],
           {"h": {2: "F"}, "noise": {2: "F10"}, "wav": {1: "T"}}, f"{out}/re_vocoder.onnx")


if __name__ == "__main__":
    main()
