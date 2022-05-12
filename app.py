import math
import tempfile
from typing import Optional, Tuple, Union

import gradio
import gradio.inputs
import gradio.outputs
import markdown
import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from torch import Tensor
from torchaudio.backend.common import AudioMetaData

from df import config
from df.enhance import enhance, init_df, load_audio, save_audio
from df.utils import resample

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, df, _ = init_df(config_allow_defaults=True)
model = model.to(device=device).eval()

NOISES = {
    "None": None,
    "Kitchen": "samples/dkitchen.wav",
    "Living Room": "samples/dliving.wav",
    "River": "samples/nriver.wav",
    "Cafe": "samples/scafe.wav",
}


def mix_at_snr(clean, noise, snr, eps=1e-10):
    """Mix clean and noise signal at a given SNR.

    Args:
        clean: 1D Tensor with the clean signal to mix.
        noise: 1D Tensor of shape.
        snr: Signal to noise ratio.

    Returns:
        clean: 1D Tensor with gain changed according to the snr.
        noise: 1D Tensor with the combined noise channels.
        mix: 1D Tensor with added clean and noise signals.

    """
    clean = torch.as_tensor(clean).mean(0, keepdim=True)
    noise = torch.as_tensor(noise).mean(0, keepdim=True)
    if noise.shape[1] < clean.shape[1]:
        noise = noise.repeat((1, int(math.ceil(clean.shape[1] / noise.shape[1]))))
    max_start = int(noise.shape[1] - clean.shape[1])
    start = torch.randint(0, max_start, ()).item()
    logger.debug(f"start: {start}, {clean.shape}")
    noise = noise[:, start : start + clean.shape[1]]
    E_speech = torch.mean(clean.pow(2)) + eps
    E_noise = torch.mean(noise.pow(2))
    K = torch.sqrt((E_noise / E_speech) * 10 ** (snr / 10) + eps)
    noise = noise / K
    mixture = clean + noise
    logger.debug("mixture: {mixture.shape}")
    assert torch.isfinite(mixture).all()
    max_m = mixture.abs().max()
    if max_m > 1:
        logger.warning(f"Clipping detected during mixing. Reducing gain by {1/max_m}")
        clean, noise, mixture = clean / max_m, noise / max_m, mixture / max_m
    return clean, noise, mixture


def load_audio_gradio(
    audio_or_file: Union[None, str, Tuple[int, np.ndarray]], sr: int
) -> Optional[Tuple[Tensor, AudioMetaData]]:
    if audio_or_file is None:
        return None
    if isinstance(audio_or_file, str):
        if audio_or_file.lower() == "none":
            return None
        # First try default format
        audio, meta = load_audio(audio_or_file, sr)
    else:
        meta = AudioMetaData(-1, -1, -1, -1, "")
        assert isinstance(audio_or_file, (tuple, list))
        meta.sample_rate, audio_np = audio_or_file
        # Gradio documentation says, the shape is [samples, 2], but apparently sometimes its not.
        audio_np = audio_np.reshape(audio_np.shape[0], -1).T
        if audio_np.dtype == np.int16:
            audio_np = (audio_np / (1 << 15)).astype(np.float32)
        elif audio_np.dtype == np.int32:
            audio_np = (audio_np / (1 << 31)).astype(np.float32)
        audio = resample(torch.from_numpy(audio_np), meta.sample_rate, sr)
    return audio, meta


def demo_fn(
    speech_rec: Union[str, Tuple[int, np.ndarray]], speech_upl: str, noise_type: str, snr: int
):
    sr = config("sr", 48000, int, section="df")
    logger.info(
        f"Got parameters speech_rec: {speech_rec}, speech_upl: {speech_upl}, noise: {noise_type}"
    )
    noise_fn = NOISES[noise_type]
    meta = AudioMetaData(-1, -1, -1, -1, "")
    if speech_rec is None and speech_upl is None:
        sample, meta = load_audio("samples/p232_013_clean.wav", sr)
    elif speech_upl is not None:
        sample, meta = load_audio(speech_upl, sr)
    else:
        tmp = load_audio_gradio(speech_rec, sr)
        assert tmp is not None
        sample, meta = tmp
    sample = sample[..., : 10 * meta.sample_rate]  # limit to 10 seconds
    logger.info(f"Loaded sample with shape {sample.shape}")
    if noise_fn is not None:
        noise, _ = load_audio(noise_fn, sr)  # type: ignore
        logger.info(f"Loaded noise with shape {noise.shape}")
        _, _, sample = mix_at_snr(sample, noise, snr)
    logger.info("Start denoising audio")
    enhanced = enhance(model, df, sample)
    logger.info("Denoising finished")
    lim = torch.linspace(0.0, 1.0, int(sr * 0.15)).unsqueeze(0)
    lim = torch.cat((lim, torch.ones(1, enhanced.shape[1] - lim.shape[1])), dim=1)
    enhanced = enhanced * lim
    # if meta.sample_rate != sr:
    #     enhanced = resample(enhanced, sr, meta.sample_rate)
    #     noisy = resample(noisy, sr, meta.sample_rate)
    #     sr = meta.sample_rate
    noisy_fn = tempfile.NamedTemporaryFile(suffix="noisy.wav", delete=False).name
    save_audio(noisy_fn, sample, sr)
    enhanced_fn = tempfile.NamedTemporaryFile(suffix="enhanced.wav", delete=False).name
    save_audio(enhanced_fn, enhanced, sr)
    logger.info(f"saved audios: {noisy_fn}, {enhanced_fn}")
    return (
        noisy_fn,
        spec_figure(sample, sr=sr),
        enhanced_fn,
        spec_figure(enhanced, sr=sr),
    )


def specshow(
    spec,
    ax=None,
    title=None,
    xlabel=None,
    ylabel=None,
    sr=48000,
    n_fft=None,
    hop=None,
    t=None,
    f=None,
    vmin=-100,
    vmax=0,
    xlim=None,
    ylim=None,
    cmap="inferno",
):
    """Plots a spectrogram of shape [F, T]"""
    spec_np = spec.cpu().numpy() if isinstance(spec, torch.Tensor) else spec
    if ax is not None:
        set_title = ax.set_title
        set_xlabel = ax.set_xlabel
        set_ylabel = ax.set_ylabel
        set_xlim = ax.set_xlim
        set_ylim = ax.set_ylim
    else:
        ax = plt
        set_title = plt.title
        set_xlabel = plt.xlabel
        set_ylabel = plt.ylabel
        set_xlim = plt.xlim
        set_ylim = plt.ylim
    if n_fft is None:
        if spec.shape[0] % 2 == 0:
            n_fft = spec.shape[0] * 2
        else:
            n_fft = (spec.shape[0] - 1) * 2
    hop = hop or n_fft // 4
    if t is None:
        t = np.arange(0, spec_np.shape[-1]) * hop / sr
    if f is None:
        f = np.arange(0, spec_np.shape[0]) * sr // 2 / (n_fft // 2) / 1000
    im = ax.pcolormesh(
        t, f, spec_np, rasterized=True, shading="auto", vmin=vmin, vmax=vmax, cmap=cmap
    )
    if title is not None:
        set_title(title)
    if xlabel is not None:
        set_xlabel(xlabel)
    if ylabel is not None:
        set_ylabel(ylabel)
    if xlim is not None:
        set_xlim(xlim)
    if ylim is not None:
        set_ylim(ylim)
    return im


def spec_figure(
    audio: torch.Tensor,
    figsize=(15, 5),
    colorbar=False,
    colorbar_format=None,
    figure=None,
    return_im=False,
    labels=True,
    **kwargs,
) -> plt.Figure:
    audio = torch.as_tensor(audio)
    if labels:
        kwargs.setdefault("xlabel", "Time [s]")
        kwargs.setdefault("ylabel", "Frequency [Hz]")
    n_fft = kwargs.setdefault("n_fft", 1024)
    hop = kwargs.setdefault("hop", 512)
    w = torch.hann_window(n_fft, device=audio.device)
    spec = torch.stft(audio, n_fft, hop, window=w, return_complex=False)
    spec = spec.div_(w.pow(2).sum())
    spec = torch.view_as_complex(spec).abs().clamp_min(1e-12).log10().mul(10)
    kwargs.setdefault("vmax", max(0.0, spec.max().item()))

    if figure is None:
        figure = plt.figure(figsize=figsize)
        figure.set_tight_layout(True)
    if spec.dim() > 2:
        spec = spec.squeeze(0)
    im = specshow(spec, **kwargs)
    if colorbar:
        ckwargs = {}
        if "ax" in kwargs:
            if colorbar_format is None:
                if kwargs.get("vmin", None) is not None or kwargs.get("vmax", None) is not None:
                    colorbar_format = "%+2.0f dB"
            ckwargs = {"ax": kwargs["ax"]}
        plt.colorbar(im, format=colorbar_format, **ckwargs)
    if return_im:
        return im
    return figure


inputs = [
    gradio.inputs.Audio(
        label="Record your own voice",
        source="microphone",
        type="numpy",
        optional=True,
    ),
    gradio.inputs.Audio(
        label="Alternative: Upload audio sample",
        source="upload",
        type="filepath",
        optional=True,
    ),
    gradio.inputs.Dropdown(
        label="Add background noise",
        choices=list(NOISES.keys()),
        default="None",
    ),
    gradio.inputs.Dropdown(
        label="Noise Level (SNR)",
        choices=[-5, 0, 10, 20],
        default=10,
    ),
]
outputs = [
    gradio.outputs.Audio(label="Noisy"),
    gradio.outputs.Image(type="plot"),
    gradio.outputs.Audio(label="Enhanced"),
    gradio.outputs.Image(type="plot"),
]
description = "This demo denoises audio files using DeepFilterNet. Try it with your own voice!"
iface = gradio.Interface(
    fn=demo_fn,
    title="DeepFilterNet2 Demo",
    inputs=inputs,
    outputs=outputs,
    description=description,
    layout="horizontal",
    allow_flagging="never",
    article=markdown.markdown(open("usage.md").read()),
)
iface.launch(cache_examples=False, debug=True)