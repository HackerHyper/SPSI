"""Dataset + collate for SPSI training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio
import whisper
from torch.utils.data import Dataset


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


class OverlapASRDataset(Dataset):
    def __init__(
        self,
        manifest: Path,
        tokenizer,
        n_mels: int = 80,
        max_frames: int = 3000,
        sr: int = 16000,
    ):
        self.rows = load_jsonl(manifest)
        self.tokenizer = tokenizer
        self.n_mels = n_mels
        self.max_frames = max_frames
        self.sr = sr

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        r = self.rows[idx]
        wav, file_sr = torchaudio.load(r["wav_path"])
        if wav.size(0) > 1:
            wav = wav.mean(0, keepdim=True)
        if file_sr != self.sr:
            wav = torchaudio.functional.resample(wav, file_sr, self.sr)
        wav = wav.squeeze(0)
        # Whisper pad/trim to 30s
        audio = whisper.pad_or_trim(wav.numpy().astype(np.float32))
        mel = whisper.log_mel_spectrogram(audio, n_mels=self.n_mels)

        soft = np.load(r["soft_path"]).astype(np.float32)  # [T, 2]
        hard = np.load(r["hard_path"]).astype(np.float32)
        overlap = ((hard[:, 0] > 0) & (hard[:, 1] > 0)).astype(np.float32)

        # SOT tokenization with Whisper special prefix (sot_sequence handles .en vs multilingual)
        text_tokens = self.tokenizer.encode(" " + r["sot_text"].strip())
        prefix = list(self.tokenizer.sot_sequence_including_notimestamps)
        tokens = prefix + text_tokens + [self.tokenizer.eot]
        tokens = tokens[:448]  # whisper n_text_ctx safety

        return {
            "mel": mel,  # [n_mels, T]
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "posterior": torch.from_numpy(soft),
            "hard_mask": torch.from_numpy(hard),
            "overlap_mask": torch.from_numpy(overlap),
            "mix_id": r["mix_id"],
            "text_spk0": r["text_spk0"],
            "text_spk1": r["text_spk1"],
            "sot_text": r["sot_text"],
        }


def collate_fn(batch: list[dict]) -> dict:
    mels = torch.stack([b["mel"] for b in batch], dim=0)
    max_len = max(b["tokens"].numel() for b in batch)
    tokens = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        tokens[i, : b["tokens"].numel()] = b["tokens"]
        # whisper CE should not ignore prompt positions only — keep all valid ids
        # replace pad -100 already; valid tokens stay

    max_t = max(b["posterior"].size(0) for b in batch)
    S = batch[0]["posterior"].size(1)
    posterior = torch.zeros(len(batch), max_t, S)
    hard = torch.zeros(len(batch), max_t, S)
    overlap = torch.zeros(len(batch), max_t)
    for i, b in enumerate(batch):
        t = b["posterior"].size(0)
        posterior[i, :t] = b["posterior"]
        hard[i, :t] = b["hard_mask"]
        overlap[i, :t] = b["overlap_mask"]

    return {
        "mel": mels,
        "tokens": tokens,
        "posterior": posterior,
        "hard_mask": hard,
        "overlap_mask": overlap,
        "mix_id": [b["mix_id"] for b in batch],
        "refs": [(b["text_spk0"], b["text_spk1"]) for b in batch],
        "sot_text": [b["sot_text"] for b in batch],
    }
