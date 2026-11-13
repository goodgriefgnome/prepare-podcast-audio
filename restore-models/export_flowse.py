"""Exports FlowSE (Honee-W/FlowSE, WenetSpeech4TTS checkpoint) and its Vocos vocoder to ONNX.

  flowse_mel.onnx       wav (1, L) 24 kHz -> mel (1, N, 100)         (log mel, as FlowSE's MelSpec)
  flowse_velocity.onnx  x (B, N, 100), cond (B, N, 100), text (B, N) int64, t (B,) -> v (B, N, 100)
                        text holds vocabulary index + 1, 0 for padding; the CFG "null" pass is
                        cond = 0, text = 0.
  flowse_vocoder.onnx   mel (1, N, 100) -> wav (1, 256 (N - 1)) 24 kHz   (charactr/vocos-mel-24khz)
  flowse_vocab.txt      the character/pinyin vocabulary (line i = token i)

The Euler ODE loop (32 steps, classifier-free guidance 1.0, as FlowSE's defaults)
runs in JavaScript.

usage: FLOWSE_SRC=<Honee-W/FlowSE checkout> python export_flowse.py <checkpoint .pt.tar> <vocos dir> <out dir> [test.wav]
"""
import math, os, shutil, sys
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn

torch.set_grad_enabled(False)
SRC = os.environ["FLOWSE_SRC"]
sys.path.insert(0, SRC)
from model import CFM, DiT  # noqa: E402
from model.model_utils import get_tokenizer, list_str_to_idx  # noqa: E402


def hann(n):
    return torch.hann_window(n, periodic=True, dtype=torch.float64)


class ConvSTFT(nn.Module):
    """torch.stft(center=True, reflect padding, Hann window) as a strided convolution."""

    def __init__(self, n_fft, hop):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        k = torch.arange(n_fft // 2 + 1, dtype=torch.float64)[:, None]
        n = torch.arange(n_fft, dtype=torch.float64)[None, :]
        ang = 2 * math.pi * k * n / n_fft
        w = hann(n_fft)
        basis = torch.cat([torch.cos(ang) * w, -torch.sin(ang) * w], 0)
        self.register_buffer("basis", basis[:, None, :].float())

    def forward(self, x):
        p = self.n_fft // 2
        x = F.pad(x[:, None, :], (p, p), mode="reflect")
        s = F.conv1d(x, self.basis, stride=self.hop)
        f = self.n_fft // 2 + 1
        return s[:, :f], s[:, f:]


class ConvISTFT(nn.Module):
    """torch.istft(center=True, no length) with a Hann window, via a transposed convolution."""

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
        c[-1] = 1
        basis = torch.cat([c * torch.cos(ang), -c * torch.sin(ang)], 0) / n_fft * w
        self.register_buffer("basis", basis[:, None, :].float())
        self.register_buffer("wsq", (w * w)[None, None, :].float())

    def forward(self, re, im):
        y = F.conv_transpose1d(torch.cat([re, im], 1), self.basis, stride=self.hop)
        env = F.conv_transpose1d(torch.ones_like(re[:, :1, :]), self.wsq, stride=self.hop)
        p = self.n_fft // 2
        return y[:, 0, p:-p] / torch.clamp(env[:, 0, p:-p], min=1e-11)


class Mel(nn.Module):
    def __init__(self, mel_spec):
        super().__init__()
        import torchaudio
        ms = torchaudio.transforms.MelSpectrogram(sample_rate=mel_spec.target_sample_rate, n_fft=mel_spec.n_fft,
                                                  win_length=mel_spec.win_length, hop_length=mel_spec.hop_length,
                                                  n_mels=mel_spec.n_mel_channels, power=1, center=True,
                                                  normalized=False, norm=None)
        assert mel_spec.win_length == mel_spec.n_fft
        self.stft = ConvSTFT(mel_spec.n_fft, mel_spec.hop_length)
        self.register_buffer("fb", ms.mel_scale.fb.T.contiguous())  # (mels, f)

    def forward(self, wav):  # (1, L) -> (1, N, mels)
        re, im = self.stft(wav)
        mag = torch.sqrt(re * re + im * im)
        mel = torch.matmul(self.fb, mag)
        return torch.log(torch.clamp(mel, min=1e-5)).transpose(1, 2)


class Velocity(nn.Module):
    def __init__(self, dit):
        super().__init__()
        self.dit = dit

    def forward(self, x, cond, text, t):
        d = self.dit
        te = d.text_embed
        seq_len = x.shape[1]
        emb = te.text_embed(text)
        if te.extra_modeling:
            pos = torch.clamp(torch.arange(seq_len), max=te.precompute_max_pos - 1)
            emb = emb + te.freqs_cis[pos][None]
            emb = te.text_blocks(emb)
        h = d.input_embed(x, cond, emb)
        rope = d.rotary_embed.forward_from_seq_len(seq_len)
        residual = h
        tt = d.time_embed(t)
        for block in d.transformer_blocks:
            h = block(h, tt, mask=None, rope=rope)
        if d.long_skip_connection is not None:
            h = d.long_skip_connection(torch.cat((h, residual), dim=-1))
        h = d.norm_out(h, tt)
        return d.proj_out(h)


class Vocoder(nn.Module):
    def __init__(self, vocos):
        super().__init__()
        self.backbone = vocos.backbone
        self.out = vocos.head.out
        self.istft = ConvISTFT(vocos.head.istft.n_fft, vocos.head.istft.hop_length)

    def forward(self, mel):  # (1, N, 100)
        x = self.backbone(mel.transpose(1, 2))
        x = self.out(x).transpose(1, 2)
        mag, p = x.chunk(2, dim=1)
        mag = torch.clip(torch.exp(mag), max=1e2)
        return self.istft(mag * torch.cos(p), mag * torch.sin(p))


def load(sd_path, vocos_dir):
    with open(f"{SRC}/config/train.yaml") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    m = conf["model"]
    vocab, vocab_size = get_tokenizer(None, m["tokenizer"], f"{SRC}/{m['tokenizer_path']}")
    nnet = CFM(transformer=DiT(**m["arch"], text_num_embeds=vocab_size, mel_dim=m["mel_spec"]["n_mel_channels"]),
               mel_spec_kwargs=m["mel_spec"], vocab_char_map=vocab).eval()
    ck = torch.load(sd_path, map_location="cpu", mmap=True, weights_only=True)
    nnet.load_state_dict(ck.get("model_state_dict", ck))  # the release also holds optimiser state
    from vocos import Vocos
    vocos = Vocos.from_hparams(f"{vocos_dir}/config.yaml")
    vocos.load_state_dict(torch.load(f"{vocos_dir}/pytorch_model.bin", map_location="cpu", weights_only=True))
    return nnet, vocos.eval(), vocab


def main():
    sd_path, vocos_dir, out = sys.argv[1:4]
    nnet, vocos, vocab = load(sd_path, vocos_dir)
    mel, vel, voc = Mel(nnet.mel_spec).eval(), Velocity(nnet.transformer).eval(), Vocoder(vocos).eval()
    g = torch.Generator().manual_seed(0)

    # module rewrites against the originals
    wav = torch.randn(1, 24000 * 2, generator=g) * 0.1
    a, b = nnet.mel_spec(wav).permute(0, 2, 1), mel(wav)
    assert torch.allclose(a, b, atol=1e-3), (a - b).abs().max()
    N = a.shape[1]
    x = torch.randn(1, N, 100, generator=g)
    text = list_str_to_idx(["hello there, this is a test."], vocab)
    tt = torch.tensor([0.37])
    for drop in (False, True):
        ref = nnet.transformer(x=x, cond=a, text=text, time=tt, mask=None, drop_audio_cond=drop, drop_text=drop)
        tok = torch.zeros(1, N, dtype=torch.long)
        if not drop:
            k = min(N, text.shape[1])
            tok[0, :k] = text[0, :k] + 1
        mine = vel(x, torch.zeros_like(a) if drop else a, tok, tt)
        assert torch.allclose(ref, mine, atol=1e-4), (drop, (ref - mine).abs().max())
    y_ref = vocos.decode(a.transpose(1, 2))
    y = voc(a)
    assert y.shape == y_ref.shape and torch.allclose(y, y_ref, atol=1e-4), (y.shape, y_ref.shape)
    print("module rewrites match")

    torch.onnx.export(mel, (wav,), f"{out}/flowse_mel.onnx", input_names=["wav"], output_names=["mel"],
                      dynamic_axes={"wav": {1: "L"}, "mel": {1: "N"}}, opset_version=17, dynamo=False)
    xb, cb = x.repeat(2, 1, 1), torch.cat([a, torch.zeros_like(a)])
    tb, tb_t = torch.zeros(2, N, dtype=torch.long), torch.tensor([0.37, 0.37])
    torch.onnx.export(vel, (xb, cb, tb, tb_t), f"{out}/flowse_velocity.onnx", input_names=["x", "cond", "text", "t"],
                      output_names=["v"], dynamic_axes={"x": {0: "B", 1: "N"}, "cond": {0: "B", 1: "N"},
                                                        "text": {0: "B", 1: "N"}, "t": {0: "B"}, "v": {0: "B", 1: "N"}},
                      opset_version=17, dynamo=False)
    torch.onnx.export(voc, (a,), f"{out}/flowse_vocoder.onnx", input_names=["mel"], output_names=["wav"],
                      dynamic_axes={"mel": {1: "N"}, "wav": {1: "L"}}, opset_version=17, dynamo=False)
    shutil.copy(f"{SRC}/Emilia_ZH_EN_pinyin/vocab.txt", f"{out}/flowse_vocab.txt")

    # whole pipeline: FlowSE's own sampling (as infer.py) vs the ONNX graphs with the same noise
    import onnxruntime as ort, torchaudio
    if len(sys.argv) > 4:
        import soundfile as sf
        w16 = torch.from_numpy(sf.read(sys.argv[4], dtype="float32")[0])[None][:, :16000 * 6]
    else:
        w16 = torch.randn(1, 16000 * 3, generator=g) * 0.1
    w24 = torchaudio.transforms.Resample(16000, 24000)(w16)
    for words in (None, "hello there"):
        torch.manual_seed(1)
        if words is None:
            ref, _ = nnet.sample(cond=w24, text=[" "], drop_text=True)
        else:
            ref, _ = nnet.sample(cond=w24, text=[words])
        ref_wav = vocos.decode(ref.transpose(1, 2)).numpy()
        torch.manual_seed(1)
        mo = ort.InferenceSession(f"{out}/flowse_mel.onnx")
        vo = ort.InferenceSession(f"{out}/flowse_velocity.onnx")
        co = ort.InferenceSession(f"{out}/flowse_vocoder.onnx")
        cond = mo.run(None, {"wav": w24.numpy()})[0]
        N = cond.shape[1]
        y = torch.randn(1, N, 100).numpy()
        tok = np.zeros((2, N), dtype=np.int64)
        if words is not None:
            ids = [vocab.get(c, 0) + 1 for c in words][:N]
            tok[0, :len(ids)] = ids
        cb = np.concatenate([cond, np.zeros_like(cond)])
        steps = 32
        ts = np.linspace(0, 1, steps + 1, dtype=np.float32)
        for i in range(steps):
            v = vo.run(None, {"x": np.concatenate([y, y]), "cond": cb, "text": tok, "t": np.full(2, ts[i], np.float32)})[0]
            v = v[:1] + (v[:1] - v[1:]) * 1.0
            y = y + (ts[i + 1] - ts[i]) * v
        wav_o = co.run(None, {"mel": y.astype(np.float32)})[0]
        e1 = np.linalg.norm(y - ref.numpy()) / np.linalg.norm(ref.numpy())
        e2 = np.linalg.norm(wav_o - ref_wav) / np.linalg.norm(ref_wav)
        print(f"text={words!r}: mel rel err {e1:.2e}, wav rel err {e2:.2e}")
        assert e1 < 1e-2


if __name__ == "__main__":
    main()
