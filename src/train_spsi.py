"""Train SPSI-Whisper on overlapped mixtures.

Round-4 additions:
  - select best.pt by validation cpWER (not train loss)
  - separate LRs for Whisper backbone vs posterior/injector
  - optional init from a prior checkpoint
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import whisper
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import OverlapASRDataset, collate_fn
from src.metrics import cp_wer
from src.spsi_whisper import SPSIConfig, SPSIWhisper


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-manifest", type=str, required=True)
    p.add_argument("--dev-manifest", type=str, default="")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--whisper", type=str, default="medium")
    p.add_argument(
        "--inject-mode",
        type=str,
        default="adapter",
        choices=["none", "film", "attn", "hard_mask", "adapter", "film_ml", "ssa", "sa_dicow"],
    )
    p.add_argument("--oracle-posterior", action="store_true")
    p.add_argument(
        "--inject-hard",
        action="store_true",
        help="Binarize posteriors before FiLM/prompts (hard-mask SPSI)",
    )
    p.add_argument("--decoder-prompt", action="store_true")
    p.add_argument("--decoder-prompt-tokens", type=int, default=4)
    p.add_argument(
        "--film-layers",
        type=str,
        default="",
        help="Comma-separated encoder block indices for film_ml (empty=auto)",
    )
    p.add_argument("--overlap-loss-weight", type=float, default=1.0)
    p.add_argument("--diar-loss-weight", type=float, default=0.5)
    p.add_argument(
        "--diar-target",
        type=str,
        default="soft",
        choices=["soft", "hard"],
        help="Soft energy-ratio targets vs exclusive hard speaker ids",
    )
    p.add_argument(
        "--keep-diar-loss",
        action="store_true",
        help="Keep L_diar even if inject_mode=none (SOT + speaker CE baseline)",
    )
    p.add_argument(
        "--aux-sdctc-weight",
        type=float,
        default=0.0,
        help="Weight for speaker-distinguishable character CTC (0=off)",
    )
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5, help="Whisper backbone LR")
    p.add_argument(
        "--lr-extra",
        type=float,
        default=-1.0,
        help="LR for posterior head + injector; <0 means same as --lr",
    )
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", action="store_false", dest="bf16")
    p.add_argument(
        "--val-max-utts",
        type=int,
        default=400,
        help="Dev utts for cpWER model selection (0=skip, use train loss)",
    )
    p.add_argument(
        "--val-select-bin",
        type=str,
        default="all",
        choices=["all", "high", "mid", "low"],
        help="Select best.pt by cpWER on this overlap bin (all=overall)",
    )
    p.add_argument("--val-every-epochs", type=int, default=1)
    p.add_argument("--init-checkpoint", type=str, default="")
    p.add_argument("--decode-max-tokens", type=int, default=224)
    p.add_argument(
        "--save-every-epoch",
        action="store_true",
        help="Also write epoch{N}.pt (large); default only last.pt + best.pt",
    )
    p.add_argument(
        "--freeze-posterior",
        action="store_true",
        help="Freeze SoftPosteriorHead (keep pretrained estimate; ASR-only FT)",
    )
    return p.parse_args()


def build_optimizer(model: SPSIWhisper, lr: float, lr_extra: float):
    backbone, extra = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("whisper."):
            backbone.append(p)
        else:
            extra.append(p)
    if lr_extra < 0:
        lr_extra = lr
    groups = []
    if backbone:
        groups.append({"params": backbone, "lr": lr})
    if extra:
        groups.append({"params": extra, "lr": lr_extra})
    if not groups:
        raise RuntimeError("No trainable parameters")
    return torch.optim.AdamW(groups, weight_decay=0.01)


def _ssa_single_speaker_batch(batch: dict, tokenizer, device: str):
    """Target-speaker ASR: pick speaker k, binary activity y^k, single-spk tokens."""
    refs = batch["refs"]
    hard = batch["hard_mask"].to(device, non_blocking=True)
    bsz = hard.size(0)
    ks = torch.randint(0, 2, (bsz,), device=hard.device)
    activity = hard[torch.arange(bsz, device=hard.device), :, ks].unsqueeze(-1)
    prefix = list(tokenizer.sot_sequence_including_notimestamps)
    eot = tokenizer.eot
    seqs = []
    for i, k in enumerate(ks.tolist()):
        text = (refs[i][k] or "").strip()
        body = tokenizer.encode(" " + text) if text else []
        seqs.append((prefix + body + [eot])[:448])
    max_len = max(len(s) for s in seqs)
    tokens = torch.full((bsz, max_len), -100, dtype=torch.long, device=device)
    for i, ids in enumerate(seqs):
        tokens[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
    return tokens, activity


@torch.no_grad()
def decode_one(
    model: SPSIWhisper,
    mel: torch.Tensor,
    posterior=None,
    hard=None,
    max_tokens: int = 224,
) -> str:
    model.eval()
    return model.transcribe_mixture(
        mel, posterior=posterior, hard_mask=hard, max_tokens=max_tokens
    )


def _sample_dev_rows(
    manifest: Path, max_utts: int, seed: int, prefer_bin: str = "all"
) -> list[dict]:
    rows = [json.loads(l) for l in open(manifest) if l.strip()]
    if prefer_bin in {"high", "mid", "low"}:
        focused = [r for r in rows if r.get("overlap_bin") == prefer_bin]
        if focused:
            rows = focused
    if max_utts <= 0 or max_utts >= len(rows):
        return rows
    by = {"low": [], "mid": [], "high": [], "?": []}
    for r in rows:
        by.setdefault(r.get("overlap_bin", "?"), []).append(r)
    rng = random.Random(seed)
    if prefer_bin in {"high", "mid", "low"} and by.get(prefer_bin):
        pool = by[prefer_bin]
        return pool if max_utts >= len(pool) else rng.sample(pool, max_utts)
    out = []
    remaining = max_utts
    bins = [b for b in ("high", "mid", "low") if by.get(b)]
    for i, b in enumerate(bins):
        if i == len(bins) - 1:
            k = remaining
        else:
            k = max(1, int(round(max_utts * len(by[b]) / max(len(rows), 1))))
            k = min(k, len(by[b]), remaining)
        pick = by[b] if k >= len(by[b]) else rng.sample(by[b], k)
        out.extend(pick)
        remaining = max_utts - len(out)
        if remaining <= 0:
            break
    if len(out) > max_utts:
        out = rng.sample(out, max_utts)
    return out


@torch.no_grad()
def validate_cpwer(
    model: SPSIWhisper,
    manifest: Path,
    device: str,
    max_utts: int,
    seed: int,
    max_tokens: int,
    select_bin: str = "all",
) -> float:
    rows = _sample_dev_rows(manifest, max_utts, seed, prefer_bin=select_bin)
    scores = []
    for r in tqdm(rows, desc=f"val-cpWER[{select_bin}]", leave=False):
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
            soft if model.cfg.use_oracle_posterior else None,
            hard,
            max_tokens=max_tokens,
        )
        scores.append(cp_wer(hyp, r["text_spk0"], r["text_spk1"]))
    model.train()
    return float(np.mean(scores)) if scores else 1.0


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    film_layers = []
    if args.film_layers.strip():
        film_layers = [int(x) for x in args.film_layers.split(",") if x.strip() != ""]
    # Diar head needed whenever we estimate (non-oracle) posteriors for inject/prompt
    # or when an auxiliary speaker loss is requested without injection.
    if args.oracle_posterior and not args.keep_diar_loss:
        diar_w = 0.0
    elif args.keep_diar_loss:
        diar_w = args.diar_loss_weight
    elif args.inject_mode == "none" and not args.decoder_prompt:
        diar_w = 0.0
    elif args.inject_mode == "ssa" and not args.keep_diar_loss:
        # Speaker Targeting trains single-speaker ASR with given activity; no L_diar.
        diar_w = 0.0
    elif args.inject_mode == "sa_dicow" and not args.keep_diar_loss:
        # SA-DiCoW conditions on oracle STNO; no estimated posterior loss.
        diar_w = 0.0
    else:
        diar_w = args.diar_loss_weight
    cfg = SPSIConfig(
        whisper_name=args.whisper,
        inject_mode=args.inject_mode,
        use_oracle_posterior=args.oracle_posterior
        or args.inject_mode in {"ssa", "sa_dicow"},
        inject_hard=bool(args.inject_hard),
        overlap_loss_weight=args.overlap_loss_weight,
        diar_loss_weight=diar_w,
        diar_target=args.diar_target,
        aux_sdctc_weight=float(args.aux_sdctc_weight),
        decoder_prompt=bool(args.decoder_prompt),
        decoder_prompt_tokens=args.decoder_prompt_tokens,
        film_layers=film_layers,
    )
    model = SPSIWhisper(cfg, device=device)
    if args.init_checkpoint:
        ck = torch.load(args.init_checkpoint, map_location=device)
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        print(
            f"Loaded init ckpt {args.init_checkpoint} "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
    if args.freeze_posterior and model.posterior_head is not None:
        for p in model.posterior_head.parameters():
            p.requires_grad = False
        print("Froze SoftPosteriorHead")
        if diar_w > 0:
            print("Note: diar loss still on but head frozen; prefer --diar-loss-weight 0")
    model.to(device)
    model.train()

    train_ds = OverlapASRDataset(
        Path(args.train_manifest), model.tokenizer, n_mels=model.dims.n_mels
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    opt = build_optimizer(model, args.lr, args.lr_extra)
    params = [p for p in model.parameters() if p.requires_grad]
    use_bf16 = bool(args.bf16 and device == "cuda")
    scaler_dtype = torch.bfloat16 if use_bf16 else torch.float32

    step = 0
    best_score = 1e9  # lower is better (cpWER or train loss)
    select_by = (
        f"val_cpwer_{args.val_select_bin}"
        if (args.dev_manifest and args.val_max_utts > 0)
        else "train_loss"
    )
    metrics_path = out / "metrics.jsonl"
    log_path = out / "train.log"
    t0 = time.time()
    with open(log_path, "a", encoding="utf-8") as logf, open(
        metrics_path, "a", encoding="utf-8"
    ) as metf:
        logf.write(f"select_by={select_by} lr={args.lr} lr_extra={args.lr_extra}\n")
        logf.flush()
        for epoch in range(args.epochs):
            pbar = tqdm(train_loader, desc=f"epoch{epoch}")
            last_loss = None
            for batch in pbar:
                step += 1
                mel = batch["mel"].to(device, non_blocking=True)
                posterior = batch["posterior"].to(device, non_blocking=True)
                hard = batch["hard_mask"].to(device, non_blocking=True)
                overlap = batch["overlap_mask"].to(device, non_blocking=True)
                if args.inject_mode == "ssa":
                    tokens, ssa_act = _ssa_single_speaker_batch(
                        batch, model.tokenizer, device
                    )
                    hard = ssa_act
                else:
                    tokens = batch["tokens"].to(device, non_blocking=True)

                dec_tokens = tokens.clone()
                dec_tokens[dec_tokens < 0] = model.tokenizer.eot

                opt.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type="cuda", dtype=scaler_dtype, enabled=use_bf16
                ):
                    out_dict = model(
                        mel,
                        dec_tokens,
                        posterior=posterior,
                        hard_mask=hard,
                        overlap_mask=overlap,
                        spk_texts=batch["refs"],
                    )
                    logits = out_dict["logits"]
                    asr_loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        tokens[:, 1:].reshape(-1),
                        ignore_index=-100,
                    )
                    diar_loss = out_dict["diar_loss"]
                    ctc_loss = out_dict.get("ctc_loss", 0.0)
                    loss = asr_loss + cfg.diar_loss_weight * (
                        diar_loss if torch.is_tensor(diar_loss) else 0.0
                    ) + cfg.aux_sdctc_weight * (
                        ctc_loss if torch.is_tensor(ctc_loss) else 0.0
                    )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                last_loss = float(loss.detach())

                if step % args.log_every == 0:
                    msg = (
                        f"step={step} epoch={epoch} loss={last_loss:.4f} "
                        f"asr={float(asr_loss.detach()):.4f} "
                        f"diar={float(diar_loss.detach() if torch.is_tensor(diar_loss) else diar_loss):.4f} "
                        f"ctc={float(ctc_loss.detach() if torch.is_tensor(ctc_loss) else ctc_loss):.4f} "
                        f"time={time.time() - t0:.1f}s"
                    )
                    pbar.set_postfix_str(msg)
                    logf.write(msg + "\n")
                    logf.flush()

                if args.max_steps > 0 and step >= args.max_steps:
                    break

            ckpt = {
                "model": model.state_dict(),
                "cfg": cfg.__dict__,
                "step": step,
                "epoch": epoch,
                "args": vars(args),
            }
            if args.save_every_epoch:
                torch.save(ckpt, out / f"epoch{epoch}.pt")
            torch.save(ckpt, out / "last.pt")

            val_cp = None
            if select_by.startswith("val_cpwer") and (
                epoch + 1
            ) % args.val_every_epochs == 0:
                val_cp = validate_cpwer(
                    model,
                    Path(args.dev_manifest),
                    device,
                    args.val_max_utts,
                    args.seed + epoch,
                    args.decode_max_tokens,
                    select_bin=args.val_select_bin,
                )
                score = val_cp
                msg = (
                    f"epoch={epoch} val_cpwer_{args.val_select_bin}={val_cp:.4f} "
                    f"(n<={args.val_max_utts})"
                )
                print(msg)
                logf.write(msg + "\n")
                logf.flush()
            else:
                score = last_loss if last_loss is not None else 1e9

            metf.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "step": step,
                        "train_loss": last_loss,
                        "val_cpwer": val_cp,
                        "select_score": score,
                    }
                )
                + "\n"
            )
            metf.flush()

            if score < best_score:
                best_score = score
                ckpt["best_score"] = best_score
                ckpt["best_metric"] = select_by
                torch.save(ckpt, out / "best.pt")
                msg = f"new best {select_by}={best_score:.4f} -> best.pt"
                print(msg)
                logf.write(msg + "\n")
                logf.flush()

            if args.max_steps > 0 and step >= args.max_steps:
                break

    print(f"Done. select_by={select_by} best={best_score:.4f} -> {out}")


if __name__ == "__main__":
    main()
