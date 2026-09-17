"""Create 2-speaker overlapping mixtures with soft speaker posteriors."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf
import torch
import torchaudio


@dataclass
class MixItem:
    mix_id: str
    wav_path: str
    text_spk0: str
    text_spk1: str
    sot_text: str
    overlap_ratio: float
    duration: float
    snr_db: float


def _load_mono(path: Path, sr: int = 16000) -> torch.Tensor:
    wav, file_sr = torchaudio.load(str(path))
    if wav.size(0) > 1:
        wav = wav.mean(0, keepdim=True)
    if file_sr != sr:
        wav = torchaudio.functional.resample(wav, file_sr, sr)
    return wav.squeeze(0)


def _scan_librispeech(root: Path) -> list[tuple[Path, str, str]]:
    """Return (wav, text, speaker_id)."""
    items: list[tuple[Path, str, str]] = []
    for trans in root.rglob("*.trans.txt"):
        spk = trans.parent.parent.name
        with open(trans, "r", encoding="utf-8") as f:
            for line in f:
                utt_id, *words = line.strip().split()
                text = " ".join(words).lower()
                wav = trans.parent / f"{utt_id}.flac"
                if wav.exists() and text:
                    items.append((wav, text, spk))
    return items


def _mix_pair(
    wav0: torch.Tensor,
    wav1: torch.Tensor,
    overlap_ratio: float,
    snr_db: float,
    sr: int = 16000,
):
    """
    Returns mix, soft_posterior [T, 2], hard_mask [T, 2] at 50Hz (20ms), true_overlap_ratio.
    Layout: start with spk0 alone, then overlap, then leftover of longer stream.
    """
    # Scale wav1 to target SNR relative to wav0
    p0 = torch.mean(wav0**2).clamp_min(1e-8)
    p1 = torch.mean(wav1**2).clamp_min(1e-8)
    scale = torch.sqrt(p0 / (p1 * (10 ** (snr_db / 10.0))))
    wav1 = wav1 * scale

    # Desired overlap length
    max_ov = min(wav0.numel(), wav1.numel())
    ov = int(max_ov * overlap_ratio)
    ov = max(ov, int(0.2 * sr))  # at least 0.2s overlap when possible
    ov = min(ov, max_ov)

    # Place wav1 so that overlap region has length `ov`
    # spk0: [0, L0), spk1: [L0-ov, L0-ov+L1)
    start1 = max(wav0.numel() - ov, 0)
    total = max(wav0.numel(), start1 + wav1.numel())
    y0 = torch.zeros(total)
    y1 = torch.zeros(total)
    y0[: wav0.numel()] = wav0
    y1[start1 : start1 + wav1.numel()] = wav1
    mix = y0 + y1

    # Frame-level activity at 50 Hz
    hop = sr // 50
    n_frames = int(np.ceil(total / hop))
    soft = torch.zeros(n_frames, 2)
    hard = torch.zeros(n_frames, 2)
    for t in range(n_frames):
        a, b = t * hop, min((t + 1) * hop, total)
        e0 = float(torch.mean(y0[a:b] ** 2))
        e1 = float(torch.mean(y1[a:b] ** 2))
        # Soft energy-based posterior (oracle for training)
        eps = 1e-8
        s0 = e0 / (e0 + e1 + eps)
        s1 = e1 / (e0 + e1 + eps)
        soft[t, 0], soft[t, 1] = s0, s1
        hard[t, 0] = 1.0 if e0 > 1e-6 else 0.0
        hard[t, 1] = 1.0 if e1 > 1e-6 else 0.0

    # True overlap ratio by time
    both = ((hard[:, 0] > 0) & (hard[:, 1] > 0)).float().mean().item()
    return mix, soft, hard, both


def build_overlap_corpus(
    librispeech_root: Path,
    out_dir: Path,
    split: str,
    num_mixtures: int,
    seed: int = 0,
    sr: int = 16000,
    min_dur: float = 2.0,
    max_dur: float = 12.0,
) -> Path:
    """Build overlapping mixtures and write JSONL manifest."""
    random.seed(seed)
    np.random.seed(seed)

    split_root = librispeech_root / "LibriSpeech" / split
    if not split_root.exists():
        # allow already-extracted layout
        split_root = librispeech_root / split
    items = _scan_librispeech(split_root)
    if len(items) < 2:
        raise RuntimeError(f"Need utterances under {split_root}, found {len(items)}")

    # Filter duration
    filtered: list[tuple[Path, str, str, float]] = []
    for wav, text, spk in items:
        info = sf.info(str(wav))
        dur = info.frames / float(info.samplerate)
        if min_dur <= dur <= max_dur:
            filtered.append((wav, text, spk, dur))
    if len(filtered) < 2:
        raise RuntimeError("Not enough utterances after duration filter")

    wav_dir = out_dir / "wavs" / split
    soft_dir = out_dir / "posteriors" / split
    hard_dir = out_dir / "hard_masks" / split
    wav_dir.mkdir(parents=True, exist_ok=True)
    soft_dir.mkdir(parents=True, exist_ok=True)
    hard_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / f"{split}_overlap.jsonl"
    by_spk: dict[str, list] = {}
    for it in filtered:
        by_spk.setdefault(it[2], []).append(it)
    speakers = [s for s, v in by_spk.items() if len(v) >= 1]
    if len(speakers) < 2:
        raise RuntimeError("Need >=2 speakers")

    written = 0
    with open(manifest_path, "w", encoding="utf-8") as mf:
        attempts = 0
        while written < num_mixtures and attempts < num_mixtures * 20:
            attempts += 1
            spk0, spk1 = random.sample(speakers, 2)
            w0, t0, _, _ = random.choice(by_spk[spk0])
            w1, t1, _, _ = random.choice(by_spk[spk1])
            overlap_ratio = random.uniform(0.15, 0.75)
            snr_db = random.uniform(-5.0, 5.0)
            try:
                wav0 = _load_mono(w0, sr)
                wav1 = _load_mono(w1, sr)
                mix, soft, hard, true_ov = _mix_pair(wav0, wav1, overlap_ratio, snr_db, sr)
            except Exception:
                continue

            mix_id = f"{split}_{written:06d}"
            wav_path = wav_dir / f"{mix_id}.wav"
            soft_path = soft_dir / f"{mix_id}.npy"
            hard_path = hard_dir / f"{mix_id}.npy"
            sf.write(str(wav_path), mix.numpy(), sr)
            np.save(soft_path, soft.numpy().astype(np.float32))
            np.save(hard_path, hard.numpy().astype(np.float32))

            sot = f"<spk0> {t0} <spk1> {t1}"
            item = MixItem(
                mix_id=mix_id,
                wav_path=str(wav_path),
                text_spk0=t0,
                text_spk1=t1,
                sot_text=sot,
                overlap_ratio=float(true_ov),
                duration=float(mix.numel() / sr),
                snr_db=float(snr_db),
            )
            row = asdict(item)
            row["soft_path"] = str(soft_path)
            row["hard_path"] = str(hard_path)
            mf.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    return manifest_path


def iter_manifest(path: Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
