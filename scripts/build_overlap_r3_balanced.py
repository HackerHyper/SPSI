"""Build Round-3 overlap corpus with forced overlap-bin quotas (high ≥30%)."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_overlap_hard_r2 import collect_pool, mix_pair, overlap_bin  # noqa: E402


def sample_for_bin(bin_name: str) -> tuple[float, float, float]:
    if bin_name == "high":
        ov = random.uniform(0.55, 0.90)
    elif bin_name == "mid":
        ov = random.uniform(0.32, 0.50)
    else:
        ov = random.uniform(0.12, 0.30)
    rt60 = random.uniform(0.25, 0.75)
    noise_snr = random.uniform(5.0, 18.0)
    return ov, rt60, noise_snr


def build_with_quotas(
    split_name: str,
    hf_split: str,
    out_dir: Path,
    n_mix: int,
    seed: int,
    pool: int,
    high_frac: float,
    mid_frac: float,
):
    random.seed(seed)
    np.random.seed(seed)
    by_spk = collect_pool(hf_split, pool)
    speakers = [s for s, v in by_spk.items() if len(v) >= 1]
    if len(speakers) < 2:
        raise RuntimeError("need >=2 speakers")

    n_high = int(round(n_mix * high_frac))
    n_mid = int(round(n_mix * mid_frac))
    n_low = n_mix - n_high - n_mid
    quotas = {"high": n_high, "mid": n_mid, "low": n_low}
    counts = {"high": 0, "mid": 0, "low": 0}
    print(f"{split_name} quotas={quotas}", flush=True)

    wav_dir = out_dir / "wavs" / split_name
    soft_dir = out_dir / "posteriors" / split_name
    hard_dir = out_dir / "hard_masks" / split_name
    for d in (wav_dir, soft_dir, hard_dir):
        d.mkdir(parents=True, exist_ok=True)
    man = out_dir / f"{split_name}_overlap.jsonl"

    written = 0
    attempts = 0
    with open(man, "w", encoding="utf-8") as mf:
        pbar = tqdm(total=n_mix, desc=f"mix-{split_name}")
        while written < n_mix and attempts < n_mix * 100:
            attempts += 1
            remain = [b for b, q in quotas.items() if counts[b] < q]
            if not remain:
                break
            target_bin = random.choice(remain)
            if counts["high"] < quotas["high"] and random.random() < 0.6:
                target_bin = "high"

            spk0, spk1 = random.sample(speakers, 2)
            w0, t0 = random.choice(by_spk[spk0])
            w1, t1 = random.choice(by_spk[spk1])
            ov, rt60, noise_snr = sample_for_bin(target_bin)
            snr = random.uniform(-5, 5)
            mix, soft, hard, true_ov = mix_pair(
                w0, w1, ov, snr, rt60=rt60, noise_snr=noise_snr
            )
            ob = overlap_bin(true_ov)
            if counts[ob] >= quotas[ob]:
                continue

            mid = f"{split_name}_{written:06d}"
            wp = wav_dir / f"{mid}.wav"
            sp = soft_dir / f"{mid}.npy"
            hp = hard_dir / f"{mid}.npy"
            sf.write(str(wp), mix.numpy(), 16000)
            np.save(sp, soft.numpy().astype(np.float32))
            np.save(hp, hard.numpy().astype(np.float32))
            row = {
                "mix_id": mid,
                "wav_path": str(wp),
                "text_spk0": t0,
                "text_spk1": t1,
                "sot_text": f"<spk0> {t0} <spk1> {t1}",
                "overlap_ratio": true_ov,
                "overlap_bin": ob,
                "duration": float(mix.numel() / 16000),
                "snr_db": float(snr),
                "rt60": float(rt60),
                "noise_snr": float(noise_snr),
                "soft_path": str(sp),
                "hard_path": str(hp),
            }
            mf.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[ob] += 1
            written += 1
            pbar.update(1)
        pbar.close()
    print(f"wrote {written} -> {man} counts={counts}", flush=True)
    return man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--train-mixtures", type=int, default=12000)
    ap.add_argument("--dev-mixtures", type=int, default=1000)
    ap.add_argument("--test-mixtures", type=int, default=1000)
    ap.add_argument("--train-pool", type=int, default=12000)
    ap.add_argument("--dev-pool", type=int, default=3000)
    ap.add_argument("--test-pool", type=int, default=3000)
    ap.add_argument("--high-frac", type=float, default=0.35)
    ap.add_argument("--mid-frac", type=float, default=0.40)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    build_with_quotas(
        "train-clean-100",
        "train.100",
        out,
        args.train_mixtures,
        args.seed,
        args.train_pool,
        args.high_frac,
        args.mid_frac,
    )
    build_with_quotas(
        "dev-clean",
        "validation",
        out,
        args.dev_mixtures,
        args.seed + 1,
        args.dev_pool,
        args.high_frac,
        args.mid_frac,
    )
    build_with_quotas(
        "test-clean",
        "test",
        out,
        args.test_mixtures,
        args.seed + 2,
        args.test_pool,
        args.high_frac,
        args.mid_frac,
    )


if __name__ == "__main__":
    main()
