"""Evaluate SPSI checkpoints with overall + overlap-bin stratified cpWER."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchaudio
import whisper
from tqdm import tqdm

from src.metrics import cp_wer, orc_wer
from src.spsi_whisper import SPSIConfig, SPSIWhisper


def overlap_bin(r: float) -> str:
    # Match R3 stored labels: ρ<0.25 / [0.25, 0.45) / ≥0.45.
    # Eval prefers r["overlap_bin"] when present.
    if r < 0.25:
        return "low"
    if r < 0.45:
        return "mid"
    return "high"


@torch.no_grad()
def decode_one(model: SPSIWhisper, mel: torch.Tensor, posterior=None, hard=None) -> str:
    model.eval()
    return model.transcribe_mixture(mel, posterior=posterior, hard_mask=hard)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--out-json", type=str, required=True)
    ap.add_argument("--max-utts", type=int, default=-1)
    ap.add_argument(
        "--force-inject",
        type=str,
        default="ckpt",
        choices=["ckpt", "oracle", "estimated"],
        help="Override checkpoint inject source: oracle P* vs estimated head",
    )
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    raw_cfg = dict(ckpt["cfg"])
    # Forward-compat: drop unknown keys if older trainers added extras
    import dataclasses

    valid = {f.name for f in dataclasses.fields(SPSIConfig)}
    cfg = SPSIConfig(**{k: v for k, v in raw_cfg.items() if k in valid})
    if args.force_inject == "oracle":
        cfg.use_oracle_posterior = True
    elif args.force_inject == "estimated":
        cfg.use_oracle_posterior = False
    model = SPSIWhisper(cfg, device=device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device)

    rows = []
    with open(args.manifest) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if args.max_utts > 0:
        rows = rows[: args.max_utts]

    by_bin = defaultdict(list)
    by_cond = defaultdict(list)
    details = []
    for r in tqdm(rows, desc="eval"):
        wav, sr = torchaudio.load(r["wav_path"])
        if wav.size(0) > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        audio = whisper.pad_or_trim(wav.squeeze(0).numpy().astype(np.float32))
        mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(device)
        soft = torch.from_numpy(np.load(r["soft_path"])).float().to(device)
        hard = torch.from_numpy(np.load(r["hard_path"])).float().to(device)
        hyp = decode_one(
            model,
            mel,
            soft if cfg.use_oracle_posterior else None,
            hard,
        )
        c = cp_wer(hyp, r["text_spk0"], r["text_spk1"])
        o = orc_wer(hyp, r["text_spk0"], r["text_spk1"])
        ob = r.get("overlap_bin") or overlap_bin(float(r.get("overlap_ratio", 0.0)))
        by_bin[ob].append(c)
        cond = r.get("condition") or "na"
        by_cond[cond].append(c)
        details.append(
            {
                "id": r["mix_id"],
                "hyp": hyp,
                "cpwer": c,
                "orcwer": o,
                "overlap_ratio": r.get("overlap_ratio"),
                "overlap_bin": ob,
                "condition": cond,
            }
        )

    all_cp = [d["cpwer"] for d in details]
    all_orc = [d["orcwer"] for d in details]
    stratified = {
        b: {"n": len(v), "cpwer": float(np.mean(v)) if v else None}
        for b, v in sorted(by_bin.items())
    }
    by_condition = {
        b: {"n": len(v), "cpwer": float(np.mean(v)) if v else None}
        for b, v in sorted(by_cond.items())
    }
    summary = {
        "n": len(all_cp),
        "cpwer": float(np.mean(all_cp)) if all_cp else None,
        "orcwer": float(np.mean(all_orc)) if all_orc else None,
        "by_overlap_bin": stratified,
        "by_condition": by_condition,
        "checkpoint": args.checkpoint,
        "manifest": args.manifest,
    }
    out = {"summary": summary, "details": details}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
