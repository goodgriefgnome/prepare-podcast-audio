"""Exports ClearerVoice's MossFormer2_SE_48K (Alibaba, Apache-2.0) to ONNX.

  mossformer2_se_48k.onnx  wav (1, 192000): 4 s at 48 kHz, -1..1 -> wav (1, 192000)

One graph does what ClearerVoice's decode does for each 4 s window: Kaldi
filterbank (60 bands) with deltas, the MossFormer2 mask network, then the
mask applied to the window's STFT and the inverse STFT. The STFTs are
convolutions. The window size is fixed: it's what ClearerVoice uses for
audio longer than 20 s (4 s windows, 3 s apart, 0.5 s discarded at each
inner edge); the windowing and stitching are done in JavaScript.

One difference: ClearerVoice adds Kaldi's default dither (random noise of
about one 16-bit step) before the filterbank; here it's off, so results are
repeatable.

usage: CV_SRC=<ClearerVoice-Studio checkout> python export_clearvoice.py <checkpoint .pt> <out dir> [test.wav]
"""
import math, os, sys, types
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch import nn

torch.set_grad_enabled(False)
# Load the model's modules without clearvoice/__init__.py, which imports the video tools.
_pkg = types.ModuleType("clearvoice")
_pkg.__path__ = [os.path.join(os.environ["CV_SRC"], "clearvoice", "clearvoice")]
sys.modules["clearvoice"] = _pkg
from clearvoice.models.mossformer2_se.mossformer2_se_wrapper import MossFormer2_SE_48K  # noqa: E402
import clearvoice.utils.decode as cv_decode  # noqa: E402

SR, WIN, HOP, NMEL = 48000, 1920, 384, 60
WINDOW = 4 * SR
MAX_WAV = 32768.0


def hamming(n):
    return torch.hamming_window(n, periodic=False, dtype=torch.float64)


class Fbank(nn.Module):
    """torchaudio.compliance.kaldi.fbank as ClearerVoice calls it (dither 0): per frame,
    remove the mean, pre-emphasise (0.97), Hamming window, 2048-point power spectrum,
    60 Kaldi mel bands, log. The per-frame steps are linear, so they fold into one
    convolution basis."""

    def __init__(self):
        super().__init__()
        N, P = WIN, 2048
        A = torch.eye(N, dtype=torch.float64) - 1.0 / N  # remove the mean
        pre = torch.eye(N, dtype=torch.float64)
        pre[1:, :-1] -= 0.97 * torch.eye(N - 1, dtype=torch.float64)
        pre[0, 0] = 1 - 0.97
        A = torch.diag(hamming(N)) @ pre @ A
        k = torch.arange(P // 2 + 1, dtype=torch.float64)[:, None]
        n = torch.arange(N, dtype=torch.float64)[None, :]
        ang = 2 * math.pi * k * n / P
        basis = torch.cat([torch.cos(ang) @ A, -torch.sin(ang) @ A], 0)
        self.register_buffer("keep32_fbank_basis", basis[:, None, :].float())
        banks, _ = torchaudio.compliance.kaldi.get_mel_banks(NMEL, P, float(SR), 20.0, 0.0, 100.0, -500.0, 1.0)
        banks = F.pad(banks, (0, 1))  # (60, 1025)
        self.register_buffer("keep32_mel", banks.float())
        self.eps = torch.finfo(torch.float).eps

    def forward(self, x):  # (1, L), 16-bit scale -> (1, T, 60)
        s = F.conv1d(x[:, None, :], self.keep32_fbank_basis, stride=HOP)
        f = s.shape[1] // 2
        power = s[:, :f] ** 2 + s[:, f:] ** 2
        mel = torch.matmul(self.keep32_mel, power)
        return torch.log(torch.clamp(mel, min=self.eps)).transpose(1, 2)


def deltas(x):
    """torchaudio.functional.compute_deltas(win_length=5) along time: x (1, T, C)."""
    p = torch.cat([x[:, :1], x[:, :1], x, x[:, -1:], x[:, -1:]], 1)
    T = x.shape[1]
    return (-2 * p[:, 0:T] - p[:, 1:T + 1] + p[:, 3:T + 3] + 2 * p[:, 4:T + 4]) / 10


class STFT(nn.Module):
    """torch.stft / torch.istft with a symmetric Hamming window, center=False."""

    def __init__(self):
        super().__init__()
        N = WIN
        w = hamming(N)
        f = N // 2 + 1
        k = torch.arange(f, dtype=torch.float64)[:, None]
        n = torch.arange(N, dtype=torch.float64)[None, :]
        ang = 2 * math.pi * k * n / N
        self.register_buffer("keep32_stft", torch.cat([torch.cos(ang) * w, -torch.sin(ang) * w], 0)[:, None, :].float())
        c = torch.full((f, 1), 2.0, dtype=torch.float64)
        c[0] = 1
        c[-1] = 1
        inv = torch.cat([c * torch.cos(ang), -c * torch.sin(ang)], 0) / N * w
        self.register_buffer("keep32_istft", inv[:, None, :].float())
        self.register_buffer("keep32_wsq", (w * w)[None, None, :].float())

    def forward(self, x, mask):  # x (1, L); mask (1, T, 961)
        s = F.conv1d(x[:, None, :], self.keep32_stft, stride=HOP)
        m = mask.transpose(1, 2)
        y = F.conv_transpose1d(s * torch.cat([m, m], 1), self.keep32_istft, stride=HOP)
        env = F.conv_transpose1d(torch.ones_like(m[:, :1]), self.keep32_wsq, stride=HOP)
        return (y / env)[:, 0]


class Window(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net, self.fbank, self.stft = net, Fbank(), STFT()

    def forward(self, wav):
        x = wav * MAX_WAV
        fb = self.fbank(x)
        d1 = deltas(fb)
        feats = torch.cat([fb, d1, deltas(d1)], 2)
        mask = self.net(feats)[-1]
        return self.stft(x, mask) / MAX_WAV


def main():
    ckpt, out = sys.argv[1:3]
    net = MossFormer2_SE_48K(None).model
    net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    net.eval()
    for m in net.modules():
        if type(m).__name__ == "RotaryEmbedding":
            m.cache_if_possible = False  # its cache check is data-dependent Python
    win = Window(net).eval()

    # ClearerVoice's own decode, with dither off, as the reference.
    def fbank_nodither(audio_in, args):
        return torchaudio.compliance.kaldi.fbank(audio_in, dither=0.0, frame_length=args.win_len / args.sampling_rate * 1000,
                                                 frame_shift=args.win_inc / args.sampling_rate * 1000, num_mel_bins=args.num_mels,
                                                 sample_frequency=args.sampling_rate, window_type=args.win_type)
    cv_decode.compute_fbank = fbank_nodither
    # one_time_decode_length 0 makes it use its windowed path, which the graph implements.
    args = types.SimpleNamespace(sampling_rate=SR, one_time_decode_length=0, decode_window=4, win_type="hamming",
                                 win_len=WIN, win_inc=HOP, fft_len=WIN, num_mels=NMEL)

    if len(sys.argv) > 3:
        import soundfile as sf, librosa
        a, sr = sf.read(sys.argv[3], dtype="float32")
        a = librosa.resample(a, orig_sr=sr, target_sr=SR)
    else:
        a = (torch.randn(SR * 5, generator=torch.Generator().manual_seed(0)) * 0.05).numpy()
    test = np.tile(a, int(np.ceil(11 * SR / len(a))))[: 11 * SR].astype(np.float32)  # 11 s: several windows

    # One window, the graph's modules against ClearerVoice's steps.
    seg = torch.from_numpy(test[:WINDOW])[None]
    x = seg[0].double() * MAX_WAV
    fb = fbank_nodither(x.float()[None], args)
    mine = win.fbank(seg * MAX_WAV)[0]
    e = (fb - mine).norm() / fb.norm()
    print(f"filterbank vs torchaudio: rel err {e:.2e}")
    assert e < 1e-4

    ref = cv_decode.decode_one_audio_mossformer2_se_48k(net, "cpu", test[None].astype(np.float64), args)

    def stitch(run, x):
        """ClearerVoice's windowed decode (window 4 s, stride 3 s, 0.5 s given up at inner edges)."""
        stride, give = WINDOW * 3 // 4, WINDOW // 8
        t = len(x)
        if t < WINDOW:
            x = np.concatenate([x, np.zeros(WINDOW - t, np.float32)])
        elif t < WINDOW + stride:
            x = np.concatenate([x, np.zeros(WINDOW + stride - t, np.float32)])
        elif (t - WINDOW) % stride != 0:
            x = np.concatenate([x, np.zeros(t - (t - WINDOW) // stride * stride, np.float32)])
        y = np.zeros(len(x), np.float32)
        i = 0
        while i + WINDOW <= len(x):
            o = run(x[i:i + WINDOW])
            if i == 0:
                y[: WINDOW - give] = o[: WINDOW - give]
            else:
                y[i + give: i + WINDOW - give] = o[give: WINDOW - give]
            i += stride
        return y

    mine = stitch(lambda s: win(torch.from_numpy(s)[None])[0].numpy(), test)
    n = min(len(ref), len(mine))
    e = np.linalg.norm(ref[:n] - mine[:n]) / np.linalg.norm(ref[:n])
    print(f"graph modules vs ClearerVoice decode: rel err {e:.2e} ({n} samples)")
    assert e < 1e-3

    torch.onnx.export(win, (seg,), f"{out}/mossformer2_se_48k.onnx", input_names=["wav"], output_names=["out"],
                      opset_version=17, dynamo=False)
    import onnxruntime as ort
    so = ort.InferenceSession(f"{out}/mossformer2_se_48k.onnx")
    mine = stitch(lambda s: so.run(None, {"wav": s[None]})[0][0], test)
    e = np.linalg.norm(ref[:n] - mine[:n]) / np.linalg.norm(ref[:n])
    print(f"ONNX vs ClearerVoice decode: rel err {e:.2e}")
    assert e < 1e-3


if __name__ == "__main__":
    main()
