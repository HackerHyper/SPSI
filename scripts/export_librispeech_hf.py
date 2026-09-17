"""Export HF LibriSpeech splits into classic LibriSpeech folder layout."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm


def export_split(split: str, out_root: Path, max_items: int = -1):
    ds = load_dataset("librispeech_asr", "clean", split=split)
    # Map HF split names to classic folder names
    name_map = {
        "train.100": "train-clean-100",
        "validation": "dev-clean",
        "test": "test-clean",
    }
    folder = name_map.get(split, split.replace(".", "-"))
    root = out_root / "LibriSpeech" / folder
    root.mkdir(parents=True, exist_ok=True)

    # group by speaker-chapter for trans.txt
    buckets: dict[tuple[str, str], list] = {}
    n = len(ds) if max_items < 0 else min(len(ds), max_items)
    for i in tqdm(range(n), desc=f"export-{split}"):
        ex = ds[i]
        # fields: file, audio, text, speaker_id, chapter_id, id
        spk = str(ex.get("speaker_id", ex.get("speaker_id", "spk")))
        chap = str(ex.get("chapter_id", "chap"))
        utt = str(ex.get("id", f"{spk}-{chap}-{i}"))
        text = str(ex["text"]).upper()
        audio = ex["audio"]
        arr = np.asarray(audio["array"], dtype=np.float32)
        sr = int(audio["sampling_rate"])
        key = (spk, chap)
        buckets.setdefault(key, []).append((utt, text, arr, sr))

    for (spk, chap), items in tqdm(buckets.items(), desc=f"write-{split}"):
        d = root / spk / chap
        d.mkdir(parents=True, exist_ok=True)
        lines = []
        for utt, text, arr, sr in items:
            # sanitize utt id for filename
            safe = utt.replace("/", "-")
            sf.write(str(d / f"{safe}.flac"), arr, sr)
            lines.append(f"{safe} {text}")
        with open(d / f"{spk}-{chap}.trans.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    print(f"Wrote {n} utts -> {root}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=str, required=True)
    ap.add_argument("--splits", type=str, default="train.100,validation,test")
    ap.add_argument("--max-items", type=int, default=-1)
    args = ap.parse_args()
    out = Path(args.out_root)
    for sp in args.splits.split(","):
        export_split(sp.strip(), out, args.max_items)


if __name__ == "__main__":
    main()
