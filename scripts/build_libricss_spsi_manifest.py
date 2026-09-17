"""Build SPSI-compatible 2-spk eval windows from LibriCSS monaural segments.

For each GT segment (all_res.json), cut <=30s chunks (Whisper limit), keep the
two most-active speakers in the chunk, write wav + soft/hard posteriors @50Hz,
and emit jsonl with condition tags (0S/0L/OV10/.../OV40).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
FPS = 50  # soft/hard frame rate used in synthetic overlap data
MAX_SEC = 30.0
MAX_SAMPLES = int(MAX_SEC * SR)

COND_RE = re.compile(r"overlap_ratio_([0-9.]+)_sil([0-9.]+)_([0-9.]+)_")


def condition_of(meeting: str) -> str:
    m = COND_RE.search(meeting)
    if not m:
        return "unknown"
    ov = float(m.group(1))
    sil0 = float(m.group(2))
    if ov == 0.0:
        return "0L" if sil0 >= 1.0 else "0S"
    return f"OV{int(ov)}"


def overlap_bin_from_ratio(r: float) -> str:
    if r < 0.30:
        return "low"
    if r < 0.50:
        return "mid"
    return "high"


def load_scp(path: Path) -> list[tuple[int, int, str, str]]:
    out = []
    if not path.exists():
        return out
    for line in open(path, encoding="utf-8", errors="ignore"):
        line = line.rstrip("\n")
        if not line:
            continue
        a, b, spk, txt = line.split("\t", 3)
        out.append((int(a), int(b), spk, txt.strip()))
    return out


def masks_for_pair(
    utts: list[tuple[int, int, str, str]],
    spk0: str,
    spk1: str,
    c0: int,
    c1: int,
) -> tuple[np.ndarray, np.ndarray, str, str, float]:
    n_frames = max(1, int(round((c1 - c0) / SR * FPS)))
    hard = np.zeros((n_frames, 2), dtype=np.float32)
    texts: dict[str, list[tuple[int, str]]] = {spk0: [], spk1: []}
    for a, b, spk, txt in utts:
        if spk not in (spk0, spk1):
            continue
        aa, bb = max(a, c0), min(b, c1)
        if bb <= aa:
            continue
        si = int((aa - c0) / SR * FPS)
        ei = int(np.ceil((bb - c0) / SR * FPS))
        si = max(0, min(n_frames, si))
        ei = max(0, min(n_frames, ei))
        col = 0 if spk == spk0 else 1
        hard[si:ei, col] = 1.0
        texts[spk].append((aa, txt))
    soft = hard.copy()  # oracle soft = hard activity; head still estimates at eval
    # Jaccard overlap of the two activity tracks
    both = float(((hard[:, 0] > 0) & (hard[:, 1] > 0)).sum())
    union = float(((hard[:, 0] > 0) | (hard[:, 1] > 0)).sum())
    ov = both / union if union > 0 else 0.0
    t0 = " ".join(t for _, t in sorted(texts[spk0]))
    t1 = " ".join(t for _, t in sorted(texts[spk1]))
    return soft, hard, t0, t1, ov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--release-root",
        type=str,
        default="/mnt/disk_4/ASR_overlap/data/libricss/data-orig/for_release",
    )
    ap.add_argument(
        "--mono-segments",
        type=str,
        default="/mnt/disk_4/ASR_overlap/data/libricss/data/monaural/segments",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="/home/zjlab/ASR/manifests/libricss",
    )
    ap.add_argument("--min-speech-sec", type=float, default=0.5)
    args = ap.parse_args()

    release = Path(args.release_root)
    mono = Path(args.mono_segments)
    out = Path(args.out_dir)
    wav_dir = out / "wavs"
    soft_dir = out / "posteriors"
    hard_dir = out / "hard_masks"
    for d in (wav_dir, soft_dir, hard_dir):
        d.mkdir(parents=True, exist_ok=True)

    all_res = json.load(open(release / "all_res.json"))
    manifest = out / "eval_2spk.jsonl"
    n = 0
    skipped = 0
    with open(manifest, "w", encoding="utf-8") as mf:
        for cond in ("0S", "0L", "OV10", "OV20", "OV30", "OV40"):
            for meet_dir in sorted((release / cond).glob("overlap*")):
                meeting = meet_dir.name
                segs = all_res[meeting]
                for si, (ss, se) in enumerate(segs):
                    scp = meet_dir / "transcription" / "segments" / f"seg_{si}.scp"
                    utts = load_scp(scp)
                    seg_wav = mono / meeting / f"segment_{si}.wav"
                    if not seg_wav.exists():
                        skipped += 1
                        continue
                    audio, sr = sf.read(str(seg_wav), always_2d=False)
                    if audio.ndim > 1:
                        audio = audio.mean(axis=-1)
                    if sr != SR:
                        raise RuntimeError(f"unexpected sr={sr} at {seg_wav}")
                    dur = len(audio)
                    off = 0
                    chunk_i = 0
                    while off < dur:
                        c0 = off
                        c1 = min(off + MAX_SAMPLES, dur)
                        # activity per speaker in this chunk (relative samples)
                        act: dict[str, int] = defaultdict(int)
                        for a, b, spk, _txt in utts:
                            aa, bb = max(a, c0), min(b, c1)
                            if bb > aa:
                                act[spk] += bb - aa
                        top = sorted(act.keys(), key=lambda s: act[s], reverse=True)
                        if (
                            len(top) >= 2
                            and act[top[0]] >= args.min_speech_sec * SR
                            and act[top[1]] >= args.min_speech_sec * SR
                        ):
                            spk0, spk1 = top[0], top[1]
                            soft, hard, t0, t1, ov = masks_for_pair(
                                utts, spk0, spk1, c0, c1
                            )
                            if t0.strip() and t1.strip():
                                mix_id = f"{meeting}__seg{si}_c{chunk_i}"
                                wav_path = wav_dir / f"{mix_id}.wav"
                                soft_path = soft_dir / f"{mix_id}.npy"
                                hard_path = hard_dir / f"{mix_id}.npy"
                                clip = np.asarray(audio[c0:c1], dtype=np.float32)
                                sf.write(str(wav_path), clip, SR)
                                np.save(soft_path, soft)
                                np.save(hard_path, hard)
                                cond_name = condition_of(meeting)
                                row = {
                                    "mix_id": mix_id,
                                    "wav_path": str(wav_path),
                                    "soft_path": str(soft_path),
                                    "hard_path": str(hard_path),
                                    "text_spk0": t0,
                                    "text_spk1": t1,
                                    "sot_text": f"<spk0> {t0} <spk1> {t1}",
                                    "overlap_ratio": ov,
                                    "overlap_bin": overlap_bin_from_ratio(ov),
                                    "condition": cond_name,
                                    "meeting": meeting,
                                    "segment_idx": si,
                                    "chunk_idx": chunk_i,
                                    "duration": (c1 - c0) / SR,
                                    "spk_ids": [spk0, spk1],
                                    "source": "libricss_monaural",
                                }
                                mf.write(json.dumps(row, ensure_ascii=False) + "\n")
                                n += 1
                        if c1 >= dur:
                            break
                        off = c1
                        chunk_i += 1

    print(f"wrote {n} windows -> {manifest} (skipped missing wavs={skipped})")


if __name__ == "__main__":
    main()
