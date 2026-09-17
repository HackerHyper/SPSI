# Soft Posterior Speaker Injection (SPSI)

Code for **Soft Posterior Speaker Injection for Multi-Talker Speech Recognition**.

Overlapped two-speaker ASR on Whisper. A Soft Posterior Head predicts a per-frame speaker share \(\hat{\mathbf{P}}\) from an unconditioned encoder pass. The share is injected with **Multi-layer Feature-wise Linear Modulation (MFLM)** and **Speaker Memory Prompts (SMP)**. No external diarizer at inference.

Paper manuscript: [`ICASSP2027__SPSI/`](ICASSP2027__SPSI/).

## Method

SPSI wraps OpenAI Whisper-medium:

| Module | Role | Code |
|--------|------|------|
| Soft Posterior Head | LN → Conv5 → GELU → Conv3 → GELU → Linear, then softmax | `SoftPosteriorHead` in [`src/spsi_whisper.py`](src/spsi_whisper.py) |
| Two-pass encoding | Pass 1: unconditioned `Enc(M)` → \(\hat{\mathbf{P}}\). Pass 2: re-encode with MFLM | `SPSIWhisper.encode` (`inject_mode="film_ml"`) |
| MFLM | FiLM after encoder blocks \(\ell \in \{8,16,24\}\) (1-based; indices `7,15,23`) | `FiLMInjector`, `MultiLayerFiLM` |
| SMP | Pool \(\hat{\mathbf{P}}\) into \(K{=}4\) prompt tokens prepended to encoder memory | `SpeakerMemoryPrompt` |

Training objective:

\[
\mathcal{L} = \mathcal{L}_{\mathrm{ASR}} + \lambda \mathcal{L}_{\mathrm{diar}},\qquad \lambda = 0.5
\]

\(\mathcal{L}_{\mathrm{ASR}}\) is teacher-forced SOT cross-entropy. \(\mathcal{L}_{\mathrm{diar}}\) is frame-level CE of \(\hat{\mathbf{P}}\) against a soft energy-ratio share on active frames.

The paper system is **`film_ml` + `--decoder-prompt`** (experiment name `spsi_film_ml_dec`).

## Results (MeetEval cpWER, sentence mean)

Controlled two-speaker LibriSpeech overlap, \(n{=}1000\) (low / mid / high = 250 / 400 / 350):

| Method | All | Low | Mid | High |
|--------|----:|----:|----:|-----:|
| SOT | 0.519 | 0.369 | 0.529 | 0.615 |
| **SPSI** | **0.510** | **0.361** | **0.524** | **0.600** |

High-overlap \(\Delta{=}1.5\) pt (\(p{=}0.034\)); full set \(\Delta{=}0.9\) pt (\(p{=}0.029\)). Paired bootstrap, \(B{=}5000\), two-sided \(H_0{:}\,\Delta_{\mathrm{cp}}{=}0\).

LibriCSS held-out sessions 8–9 (\(n{=}259\)): freeze-posterior overlap-heavy adaptation reduces cpWER from **42.3%** (SOT) to **36.8%** (SPSI).

## Repository layout

```
src/spsi_whisper.py      # model: head, MFLM, SMP, two-pass encode
src/train_spsi.py        # training
src/dataset.py           # overlap jsonl + collate
src/metrics.py           # cpWER / ORC-WER
src/overlap_mix.py       # mixing utilities
scripts/                 # data, eval, paper plots, launchers
ICASSP2027__SPSI/        # ICASSP manuscript
```

Large artifacts (`checkpoints/`, `evals/`, `logs/`, `manifests/`, `data/`) are experiment outputs. Do not commit them.

## Requirements

- Python 3.10+
- CUDA GPU (training used BF16 on a single GPU, batch size 4)
- Packages: `torch`, `torchaudio`, `openai-whisper`, `numpy`, `soundfile`, `tqdm`

```bash
pip install torch torchaudio openai-whisper numpy soundfile tqdm
```

Whisper weights download on first load. Optional cache:

```bash
export WHISPER_CACHE=/path/to/whisper/weights
```

## Setup

```bash
git clone https://github.com/HackerHyper/SPSI.git
cd SPSI
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
```

## Data

### Synthetic two-speaker overlap (LibriSpeech)

1. Export LibriSpeech (Hugging Face `librispeech_asr` or a local copy):

```bash
python scripts/export_librispeech_hf.py \
  --out-root /path/to/librispeech_export
```

2. Build the paper split (train / dev / test = 12k / 1k / 1k; high-overlap quota ≈ 35%; bins \(\rho{<}0.25\), \([0.25,0.45)\), \(\ge 0.45\)):

```bash
python scripts/build_overlap_r3_balanced.py \
  --librispeech-root /path/to/librispeech_export \
  --out-dir manifests/overlap_r3 \
  --train-mixtures 12000 --dev-mixtures 1000 --test-mixtures 1000
```

Each jsonl row has `wav_path`, SOT text (`<spk0> … <spk1> …`), and frame-level `soft_path` / `hard_path` (energy-ratio share and hard activity).

### LibriCSS (domain transfer)

Prepare two-speaker ≤30 s windows from LibriCSS monaural recordings:

```bash
python scripts/build_libricss_spsi_manifest.py \
  --libricss-root /path/to/LibriCSS \
  --out-dir manifests/libricss
```

Paper split: sessions **0–6** train (\(n{=}926\)), **7** development, **8–9** held-out test (\(n{=}259\)).

## Train (paper SPSI)

```bash
python src/train_spsi.py \
  --train-manifest manifests/overlap_r3/train-clean-100_overlap.jsonl \
  --dev-manifest manifests/overlap_r3/dev-clean_overlap.jsonl \
  --output-dir checkpoints/spsi_film_ml_dec \
  --whisper medium \
  --inject-mode film_ml \
  --decoder-prompt \
  --decoder-prompt-tokens 4 \
  --diar-loss-weight 0.5 \
  --epochs 6 \
  --batch-size 4 \
  --bf16 \
  --lr 2e-6 \
  --lr-extra 5e-5 \
  --val-max-utts 350 \
  --val-select-bin high \
  --val-every-epochs 1 \
  --seed 0
```

- `--lr` is the Whisper backbone; `--lr-extra` is the posterior head, MFLM, and SMP.
- `best.pt` is the checkpoint with the lowest **high-overlap** development cpWER.

### Ablations (same trainer)

| Paper name | Flags |
|------------|--------|
| SOT | `--inject-mode none --diar-loss-weight 0.0` |
| MFLM (no SMP) | `--inject-mode film_ml` |
| SMP (no MFLM) | `--inject-mode none --decoder-prompt --diar-loss-weight 0.5` |
| Single-layer MFLM + SMP | `--inject-mode film --decoder-prompt` |
| **SPSI** | `--inject-mode film_ml --decoder-prompt` |

Other `--inject-mode` values (`attn`, `adapter`, `hard_mask`, `ssa`, `sa_dicow`) and `--aux-sdctc-weight` implement baselines in the paper, not SPSI.

## Evaluate

Per-bin cpWER on the synthetic test set:

```bash
python scripts/eval_spsi_stratified.py \
  --checkpoint checkpoints/spsi_film_ml_dec/best.pt \
  --manifest manifests/overlap_r3/test-clean_overlap.jsonl \
  --out-json evals/spsi_film_ml_dec.json
```

The paper tables use CHiME-6 / MeetEval concatenated minimum-permutation WER (utterance mean). After decoding, you can rescore with [`scripts/rescore_cpwer_meeteval.py`](scripts/rescore_cpwer_meeteval.py) if MeetEval is installed.

## LibriCSS transfer

Keep the synthetic-trained posterior head fixed and adapt the rest (\(\lambda{=}0\)):

```bash
python src/train_spsi.py \
  --train-manifest manifests/libricss/train_2spk.jsonl \
  --dev-manifest manifests/libricss/dev_2spk.jsonl \
  --output-dir checkpoints/libricss_freeze \
  --init-checkpoint checkpoints/spsi_film_ml_dec/best.pt \
  --inject-mode film_ml --decoder-prompt \
  --freeze-posterior \
  --diar-loss-weight 0.0 \
  --whisper medium --epochs 6 --batch-size 2 --bf16 \
  --lr 5e-7 --lr-extra 2e-6 \
  --val-select-bin all --seed 0
```

Overlap-heavy continuation uses the OV20–OV40 subset of the train windows, still with `--freeze-posterior`. Jointly updating the head (`--diar-loss-weight 0.0` without freeze) overwrites the synthetic share and is worse than SOT in the paper.

## Inference sketch

```python
import torch, whisper
from src.spsi_whisper import SPSIConfig, SPSIWhisper

ckpt = torch.load("checkpoints/spsi_film_ml_dec/best.pt", map_location="cuda")
cfg = SPSIConfig(**ckpt["cfg"])
model = SPSIWhisper(cfg, device="cuda")
model.load_state_dict(ckpt["model"], strict=False)
model.eval()

audio = whisper.load_audio("mix.wav")
audio = whisper.pad_or_trim(audio)
mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to("cuda")
text = model.transcribe_mixture(mel)  # SOT string: <spk0> ... <spk1> ...
```

## Citation

```bibtex
@inproceedings{zhu2027spsi,
  title     = {Soft Posterior Speaker Injection for Multi-Talker Speech Recognition},
  author    = {Zhu, Jian and Sun, Jun and Yang, Jiang and Zhou, Ying
               and Luo, Cheng and Liu, Cong and Dai, Li-Rong},
  booktitle = {ICASSP},
  year      = {2027}
}
```

