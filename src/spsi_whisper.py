"""SPSI-Whisper: Soft Posterior Speaker Injection for overlapped ASR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import whisper
from whisper.model import ModelDimensions


def _align_time(posterior: torch.Tensor, T: int) -> torch.Tensor:
    if posterior.size(1) == T:
        return posterior
    return F.interpolate(
        posterior.transpose(1, 2),
        size=T,
        mode="linear",
        align_corners=False,
    ).transpose(1, 2)


class SoftPosteriorHead(nn.Module):
    """Temporal-aware frame-level speaker posterior predictor."""

    def __init__(self, n_state: int, n_speakers: int = 2, hidden: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(n_state)
        self.temporal = nn.Sequential(
            nn.Conv1d(n_state, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.out = nn.Linear(hidden, n_speakers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x).transpose(1, 2)
        h = self.temporal(h).transpose(1, 2)
        return self.out(h)


class FiLMInjector(nn.Module):
    """Modulate encoder features with soft speaker posteriors via FiLM."""

    def __init__(self, n_state: int, n_speakers: int = 2):
        super().__init__()
        self.to_gb = nn.Sequential(
            nn.Linear(n_speakers, n_state),
            nn.GELU(),
            nn.Linear(n_state, 2 * n_state),
        )

    def forward(self, x: torch.Tensor, posterior: torch.Tensor) -> torch.Tensor:
        posterior = _align_time(posterior, x.size(1))
        gb = self.to_gb(posterior)
        gamma, beta = gb.chunk(2, dim=-1)
        return x * (1 + torch.tanh(gamma)) + beta


class CrossAttnInjector(nn.Module):
    """Use soft posterior as value/key side info via cross-attention."""

    def __init__(self, n_state: int, n_speakers: int = 2, n_heads: int = 4):
        super().__init__()
        self.proj = nn.Linear(n_speakers, n_state)
        self.attn = nn.MultiheadAttention(n_state, n_heads, batch_first=True)
        self.out = nn.Linear(n_state, n_state)

    def forward(self, x: torch.Tensor, posterior: torch.Tensor) -> torch.Tensor:
        posterior = _align_time(posterior, x.size(1))
        spk = self.proj(posterior)
        y, _ = self.attn(x, spk, spk, need_weights=False)
        return x + self.out(y)


class SpeakerMaskAdapter(nn.Module):
    """Per-speaker soft-mask branches + gated fusion + light FiLM."""

    def __init__(self, n_state: int, n_speakers: int = 2, hidden: int = 512):
        super().__init__()
        self.n_speakers = n_speakers
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(n_state, hidden, kernel_size=1),
                    nn.GELU(),
                    nn.Conv1d(
                        hidden, hidden, kernel_size=5, padding=2, groups=min(8, hidden)
                    ),
                    nn.GELU(),
                    nn.Conv1d(hidden, n_state, kernel_size=1),
                )
                for _ in range(n_speakers)
            ]
        )
        self.alpha = nn.Sequential(
            nn.Linear(n_speakers, n_speakers * 2),
            nn.GELU(),
            nn.Linear(n_speakers * 2, n_speakers),
        )
        self.film = FiLMInjector(n_state, n_speakers)
        self.out_norm = nn.LayerNorm(n_state)
        self.gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor, posterior: torch.Tensor) -> torch.Tensor:
        posterior = _align_time(posterior, x.size(1))
        xt = x.transpose(1, 2)
        fused = 0.0
        alpha = torch.softmax(self.alpha(posterior), dim=-1)
        for s, branch in enumerate(self.branches):
            ps = posterior[:, :, s : s + 1].transpose(1, 2)
            hs = branch(xt * ps)
            a = alpha[:, :, s : s + 1].transpose(1, 2)
            fused = fused + hs * a
        fused = fused.transpose(1, 2)
        filmed = self.film(x, posterior)
        g = torch.sigmoid(self.gate)
        return self.out_norm(x + g * fused + (1 - g) * (filmed - x))


class MultiLayerFiLM(nn.Module):
    """FiLM after selected encoder blocks (deeper coupling)."""

    def __init__(
        self, n_state: int, n_speakers: int, layer_indices: List[int], n_layers: int
    ):
        super().__init__()
        self.layer_indices = sorted(set(int(i) for i in layer_indices if 0 <= i < n_layers))
        self.injectors = nn.ModuleDict(
            {str(i): FiLMInjector(n_state, n_speakers) for i in self.layer_indices}
        )

    def apply_at(self, layer_i: int, x: torch.Tensor, posterior: torch.Tensor) -> torch.Tensor:
        key = str(layer_i)
        if key not in self.injectors:
            return x
        return self.injectors[key](x, posterior)


def hard_to_stno(hard_mask: torch.Tensor, target_idx: int) -> torch.Tensor:
    """Oracle hard activity [B, T, S] → one-hot STNO [B, T, 4] for speaker ``target_idx``.

    S: silence, T: target only, N: non-target only, O: overlap (Kocour et al., ICASSP 2026).
    """
    if hard_mask.dim() == 2:
        hard_mask = hard_mask.unsqueeze(0)
    tgt = hard_mask[..., target_idx]
    others = hard_mask.sum(dim=-1) - tgt
    a_t = tgt > 0.5
    a_n = others > 0.5
    silence = (~a_t & ~a_n).to(dtype=hard_mask.dtype)
    target = (a_t & ~a_n).to(dtype=hard_mask.dtype)
    nontgt = (~a_t & a_n).to(dtype=hard_mask.dtype)
    overlap = (a_t & a_n).to(dtype=hard_mask.dtype)
    return torch.stack([silence, target, nontgt, overlap], dim=-1)


class FDDTLayer(nn.Module):
    """Frame-level diarization-dependent transformation (DiCoW / SA-DiCoW).

    ĥ_t = Σ_c (W_c h_t + b_c) p_{t,c}, c ∈ {S, T, N, O}.
    Affines are identity-initialized so the pretrained encoder is unchanged at step 0.
    """

    def __init__(self, n_state: int, n_classes: int = 4):
        super().__init__()
        self.affines = nn.ModuleList(
            [nn.Linear(n_state, n_state) for _ in range(n_classes)]
        )
        for lin in self.affines:
            nn.init.eye_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, h: torch.Tensor, stno: torch.Tensor) -> torch.Tensor:
        p = _align_time(stno, h.size(1)).to(dtype=h.dtype)
        out = 0.0
        for c, lin in enumerate(self.affines):
            out = out + lin(h) * p[..., c : c + 1]
        return out


class MultiLayerFDDT(nn.Module):
    """FDDT on the input of every encoder block."""

    def __init__(self, n_state: int, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([FDDTLayer(n_state) for _ in range(n_layers)])

    def apply_at(self, layer_i: int, x: torch.Tensor, stno: torch.Tensor) -> torch.Tensor:
        return self.layers[layer_i](x, stno)


class SpeakerChannelAffine(nn.Module):
    """Per-speaker affine on DiCoW encoder outputs, identity-initialized."""

    def __init__(self, n_state: int, n_speakers: int = 2):
        super().__init__()
        self.affines = nn.ModuleList(
            [nn.Linear(n_state, n_state) for _ in range(n_speakers)]
        )
        for lin in self.affines:
            nn.init.eye_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, h: torch.Tensor, speaker_idx: int) -> torch.Tensor:
        return self.affines[speaker_idx](h)


class SSAInjector(nn.Module):
    """Speaker Targeting / self-speaker adaptation (Wang et al., Interspeech 2025).

    Inject binary target-speaker activity at the encoder pre-block input:
        X' = f_FF(X ⊙ y) + X,  y ∈ {0,1}^{T}.
    One forward pass recognizes one speaker; inference runs twice.
    """

    def __init__(self, n_state: int):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(n_state, n_state),
            nn.GELU(),
            nn.Linear(n_state, n_state),
        )

    def forward(self, x: torch.Tensor, activity: torch.Tensor) -> torch.Tensor:
        if activity.dim() == 2:
            y = activity.unsqueeze(-1)
        elif activity.size(-1) > 1:
            y = activity[..., :1]
        else:
            y = activity
        t = x.size(1)
        if y.size(1) < t:
            y = torch.cat(
                [y, y.new_zeros(y.size(0), t - y.size(1), y.size(-1))], dim=1
            )
        elif y.size(1) > t:
            y = y[:, :t]
        gated = x * y.to(dtype=x.dtype)
        return x + self.ff(gated)


class SpeakerMemoryPrompt(nn.Module):
    """
    Prepend soft speaker summary tokens to encoder memory so the decoder
    cross-attends to explicit speaker conditioning (decoder-side coupling).
    """

    def __init__(self, n_state: int, n_speakers: int = 2, n_prompt: int = 4):
        super().__init__()
        self.n_prompt = n_prompt
        self.proj = nn.Sequential(
            nn.Linear(n_speakers * 3, n_state),
            nn.GELU(),
            nn.Linear(n_state, n_prompt * n_state),
        )
        self.norm = nn.LayerNorm(n_state)

    def forward(self, audio_features: torch.Tensor, posterior: torch.Tensor) -> torch.Tensor:
        p = _align_time(posterior, audio_features.size(1))
        mean = p.mean(dim=1)
        mx = p.amax(dim=1)
        # overlap mass: product of speaker probs as soft overlap cue
        ov = (p[:, :, 0] * p[:, :, 1]).mean(dim=1, keepdim=True).expand(-1, p.size(-1))
        h = self.proj(torch.cat([mean, mx, ov], dim=-1))
        bsz, _ = h.shape
        prompts = self.norm(h.view(bsz, self.n_prompt, audio_features.size(-1)))
        return torch.cat([prompts, audio_features], dim=1)


def default_film_layers(n_layers: int) -> List[int]:
    """Inject at ~1/3, 2/3, and final block."""
    if n_layers <= 1:
        return [0]
    a = max(0, n_layers // 3 - 1)
    b = max(a + 1, (2 * n_layers) // 3 - 1)
    c = n_layers - 1
    return sorted({a, b, c})


# Character CTC alphabet for SD-CTC-lite (blank = 0).
_SDCTC_CHARS = "abcdefghijklmnopqrstuvwxyz '"
_SDCTC_CHAR2I = {c: i + 1 for i, c in enumerate(_SDCTC_CHARS)}
SDCTC_VOCAB = 1 + len(_SDCTC_CHARS)


def encode_sdctc_text(text: str) -> list[int]:
    ids = []
    for c in (text or "").lower():
        i = _SDCTC_CHAR2I.get(c)
        if i is not None:
            ids.append(i)
    return ids


class TokenCTCHead(nn.Module):
    """Frame-level character distribution P_v(π | t) for SD-CTC."""

    def __init__(self, n_state: int, vocab: int = SDCTC_VOCAB):
        super().__init__()
        self.proj = nn.Linear(n_state, vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def sdctc_loss(
    spk_logits: torch.Tensor,
    tok_logits: torch.Tensor,
    texts: list[tuple[str, str]],
    input_lengths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Speaker-Distinguishable CTC (Sakuma et al., 2025), character-level.

    For speaker σ, non-blank tokens use P(σ) P_v(ρ); the speaker-specific
    blank <¬σ> absorbs non-speech and other-speaker frames:
        P(¬σ | t) = P(σ|t) P_v(blank|t) + (1 - P(σ|t)).
    """
    log_ps = F.log_softmax(spk_logits, dim=-1)
    log_pv = F.log_softmax(tok_logits, dim=-1)
    ps = log_ps.exp()
    pv = log_pv.exp()
    bsz, t_enc, n_spk = ps.shape
    device = ps.device
    if input_lengths is None:
        input_lengths = torch.full((bsz,), t_enc, device=device, dtype=torch.long)
    total = ps.new_zeros(())
    n_ok = 0
    for s in range(n_spk):
        p_blank = ps[:, :, s] * pv[:, :, 0] + (1.0 - ps[:, :, s])
        p_tok = ps[:, :, s : s + 1] * pv[:, :, 1:]
        dist = torch.cat([p_blank.unsqueeze(-1), p_tok], dim=-1).clamp_min(1e-8)
        logp = dist.log().transpose(0, 1).contiguous()  # [T, B, V]
        padded: list[torch.Tensor] = []
        lengths = []
        for b in range(bsz):
            ids = encode_sdctc_text(texts[b][s] if s < len(texts[b]) else "")
            if not ids:
                ids = [1]  # dummy 'a'; masked via target length 0 is invalid for CTC
                lengths.append(0)
            else:
                lengths.append(len(ids))
            padded.append(torch.tensor(ids, device=device, dtype=torch.long))
        max_u = max(p.numel() for p in padded)
        targets = torch.zeros(bsz, max_u, device=device, dtype=torch.long)
        for b, p in enumerate(padded):
            targets[b, : p.numel()] = p
        tlen = torch.tensor(lengths, device=device, dtype=torch.long)
        valid = tlen > 0
        if not bool(valid.any()):
            continue
        # CTC requires input_length >= target_length
        ilen = input_lengths.clamp(min=1)
        tlen = torch.minimum(tlen, ilen)
        loss_s = F.ctc_loss(
            logp,
            targets,
            ilen,
            tlen,
            blank=0,
            reduction="mean",
            zero_infinity=True,
        )
        if torch.isfinite(loss_s):
            total = total + loss_s
            n_ok += 1
    if n_ok == 0:
        return spk_logits.new_zeros(())
    return total / n_ok


@dataclass
class SPSIConfig:
    whisper_name: str = "medium"
    # none | film | attn | hard_mask | adapter | film_ml | ssa | sa_dicow
    inject_mode: str = "adapter"
    use_oracle_posterior: bool = False
    inject_hard: bool = False
    n_speakers: int = 2
    overlap_loss_weight: float = 1.0
    diar_loss_weight: float = 0.5
    diar_target: str = "soft"  # soft | hard
    aux_sdctc_weight: float = 0.0
    freeze_encoder: bool = False
    decoder_prompt: bool = False
    decoder_prompt_tokens: int = 4
    film_layers: List[int] = field(default_factory=list)


class SPSIWhisper(nn.Module):
    """
    Wrap OpenAI Whisper with soft speaker posterior injection.

    inject_mode:
      - none: plain SOT Whisper
      - film / hard_mask / attn / adapter: post-encoder injection
      - film_ml: FiLM after multiple encoder blocks
      - ssa: Speaker Targeting — binary activity at pre-encode, one speaker / pass
      - sa_dicow: SA-DiCoW replica — FDDT per speaker, concat channels, joint SOT
    decoder_prompt:
      - prepend speaker memory tokens to encoder features for decoder cross-attn
    """

    def __init__(self, cfg: SPSIConfig, device: str = "cuda"):
        super().__init__()
        self.cfg = cfg
        download_root = __import__("os").environ.get(
            "WHISPER_CACHE", "/mnt/disk_4/ASR_overlap/models/whisper"
        )
        self.whisper = whisper.load_model(
            cfg.whisper_name, device=device, download_root=download_root
        )
        dims: ModelDimensions = self.whisper.dims
        self.dims = dims
        self.posterior_head = SoftPosteriorHead(dims.n_audio_state, cfg.n_speakers)
        self.ctc_head: Optional[TokenCTCHead] = None
        if cfg.aux_sdctc_weight > 0:
            self.ctc_head = TokenCTCHead(dims.n_audio_state)

        mode = cfg.inject_mode
        n_enc = len(self.whisper.encoder.blocks)
        self.ml_film: Optional[MultiLayerFiLM] = None
        self.injector: Optional[nn.Module] = None
        self.ssa_injector: Optional[SSAInjector] = None
        self.fddt: Optional[MultiLayerFDDT] = None
        self.spk_channel_affine: Optional[SpeakerChannelAffine] = None

        if mode == "attn":
            self.injector = CrossAttnInjector(dims.n_audio_state, cfg.n_speakers)
        elif mode == "adapter":
            self.injector = SpeakerMaskAdapter(dims.n_audio_state, cfg.n_speakers)
        elif mode in {"film", "hard_mask"}:
            self.injector = FiLMInjector(dims.n_audio_state, cfg.n_speakers)
        elif mode == "film_ml":
            layers = cfg.film_layers or default_film_layers(n_enc)
            cfg.film_layers = list(layers)
            self.ml_film = MultiLayerFiLM(
                dims.n_audio_state, cfg.n_speakers, layers, n_enc
            )
        elif mode == "ssa":
            self.ssa_injector = SSAInjector(dims.n_audio_state)
        elif mode == "sa_dicow":
            self.fddt = MultiLayerFDDT(dims.n_audio_state, n_enc)
            self.spk_channel_affine = SpeakerChannelAffine(
                dims.n_audio_state, cfg.n_speakers
            )
        else:
            self.injector = None

        self.decoder_prompt_mod: Optional[SpeakerMemoryPrompt] = None
        if cfg.decoder_prompt:
            self.decoder_prompt_mod = SpeakerMemoryPrompt(
                dims.n_audio_state, cfg.n_speakers, cfg.decoder_prompt_tokens
            )

        if cfg.freeze_encoder:
            for p in self.whisper.encoder.parameters():
                p.requires_grad = False

        self.tokenizer = whisper.tokenizer.get_tokenizer(
            self.whisper.is_multilingual,
            num_languages=self.whisper.num_languages,
            language="en",
            task="transcribe",
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def _resolve_inj(
        self,
        pred_logits: torch.Tensor,
        posterior: Optional[torch.Tensor],
        hard_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        mode = self.cfg.inject_mode
        needs = mode != "none" or self.cfg.decoder_prompt
        if not needs:
            return None
        use_hard = mode == "hard_mask" or self.cfg.inject_hard
        if use_hard:
            if self.cfg.use_oracle_posterior and hard_mask is not None:
                return hard_mask
            # Mutually exclusive hard assignment: one speaker per frame.
            return (torch.softmax(pred_logits, dim=-1) > 0.5).float()
        if self.cfg.use_oracle_posterior and posterior is not None:
            return posterior
        return torch.softmax(pred_logits, dim=-1)

    def _encoder_forward_ml(
        self, mel: torch.Tensor, inj: torch.Tensor
    ) -> torch.Tensor:
        enc = self.whisper.encoder
        x = F.gelu(enc.conv1(mel))
        x = F.gelu(enc.conv2(x))
        x = x.permute(0, 2, 1)
        assert x.shape[1:] == enc.positional_embedding.shape, "incorrect audio shape"
        x = (x + enc.positional_embedding).to(x.dtype)
        assert self.ml_film is not None
        for i, block in enumerate(enc.blocks):
            x = block(x)
            x = self.ml_film.apply_at(i, x, inj.to(x.dtype))
        return enc.ln_post(x)

    def _encoder_forward_ssa(
        self, mel: torch.Tensor, activity: torch.Tensor
    ) -> torch.Tensor:
        enc = self.whisper.encoder
        x = F.gelu(enc.conv1(mel))
        x = F.gelu(enc.conv2(x))
        x = x.permute(0, 2, 1)
        assert x.shape[1:] == enc.positional_embedding.shape, "incorrect audio shape"
        x = (x + enc.positional_embedding).to(x.dtype)
        assert self.ssa_injector is not None
        x = self.ssa_injector(x, activity)
        for block in enc.blocks:
            x = block(x)
        return enc.ln_post(x)

    def _encoder_forward_fddt(
        self, mel: torch.Tensor, stno: torch.Tensor
    ) -> torch.Tensor:
        enc = self.whisper.encoder
        x = F.gelu(enc.conv1(mel))
        x = F.gelu(enc.conv2(x))
        x = x.permute(0, 2, 1)
        assert x.shape[1:] == enc.positional_embedding.shape, "incorrect audio shape"
        x = (x + enc.positional_embedding).to(x.dtype)
        assert self.fddt is not None
        for i, block in enumerate(enc.blocks):
            x = self.fddt.apply_at(i, x, stno)
            x = block(x)
        return enc.ln_post(x)

    def _encode_sa_dicow(
        self, mel: torch.Tensor, hard_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-speaker FDDT encode, speaker affine, time-concat (SA-DiCoW)."""
        assert self.spk_channel_affine is not None
        if hard_mask.dim() == 2:
            hard_mask = hard_mask.unsqueeze(0)
        channels = []
        for u in range(self.cfg.n_speakers):
            stno = hard_to_stno(hard_mask, u)
            hu = self._encoder_forward_fddt(mel, stno)
            hu = self.spk_channel_affine(hu, u)
            channels.append(hu)
        audio_features = torch.cat(channels, dim=1)
        pred_logits = self.posterior_head(channels[0])
        return audio_features, pred_logits

    def encode(
        self,
        mel: torch.Tensor,
        posterior: Optional[torch.Tensor] = None,
        hard_mask: Optional[torch.Tensor] = None,
        activity: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mode = self.cfg.inject_mode

        if mode == "sa_dicow":
            if hard_mask is None:
                raise ValueError("SA-DiCoW encode requires oracle hard_mask STNO")
            return self._encode_sa_dicow(mel, hard_mask)

        if mode == "ssa":
            act = activity
            if act is None and hard_mask is not None:
                if hard_mask.dim() == 3 and hard_mask.size(-1) == 1:
                    act = hard_mask.squeeze(-1)
                elif hard_mask.dim() == 3:
                    act = hard_mask[..., 0]
                else:
                    act = hard_mask
            if act is None:
                act = torch.ones(mel.size(0), 1, device=mel.device, dtype=mel.dtype)
            audio_features = self._encoder_forward_ssa(mel, act)
            pred_logits = self.posterior_head(audio_features)
            return audio_features, pred_logits

        if mode == "film_ml":
            # Preliminary features for posterior head: standard encoder once
            # would be 2x cost. Instead: run stem+blocks without inject for head,
            # then re-run with inject — too expensive.
            # Practical: predict posterior from unconditioned encoder, then
            # re-encode with FiLM using stopgrad posterior for inject path only
            # when not oracle. For speed: one pass — head on post-conv mid features
            # is messy. We do: full encoder without ml, head, then ml encoder with inj.
            base = self.whisper.encoder(mel)
            pred_logits = self.posterior_head(base)
            inj = self._resolve_inj(pred_logits, posterior, hard_mask)
            assert inj is not None
            audio_features = self._encoder_forward_ml(mel, inj)
        else:
            audio_features = self.whisper.encoder(mel)
            pred_logits = self.posterior_head(audio_features)
            inj = self._resolve_inj(pred_logits, posterior, hard_mask)
            if mode != "none" and self.injector is not None and inj is not None:
                audio_features = self.injector(
                    audio_features, inj.to(audio_features.dtype)
                )

        if self.decoder_prompt_mod is not None:
            inj_p = inj if inj is not None else self._resolve_inj(
                pred_logits, posterior, hard_mask
            )
            if inj_p is not None:
                audio_features = self.decoder_prompt_mod(
                    audio_features, inj_p.to(audio_features.dtype)
                )

        return audio_features, pred_logits

    @torch.no_grad()
    def greedy_decode(self, audio_features: torch.Tensor, max_tokens: int = 224) -> str:
        tokenizer = self.tokenizer
        prefix = list(tokenizer.sot_sequence_including_notimestamps)
        device = audio_features.device
        tokens = torch.tensor([prefix], device=device)
        for _ in range(max_tokens):
            logits = self.whisper.decoder(tokens, audio_features)
            next_id = int(logits[0, -1].argmax())
            if next_id == tokenizer.eot:
                break
            tokens = torch.cat(
                [tokens, torch.tensor([[next_id]], device=device)], dim=1
            )
        return tokenizer.decode(tokens[0, len(prefix) :].tolist())

    @torch.no_grad()
    def transcribe_mixture(
        self,
        mel: torch.Tensor,
        posterior: Optional[torch.Tensor] = None,
        hard_mask: Optional[torch.Tensor] = None,
        max_tokens: int = 224,
    ) -> str:
        """Unbatched mel [n_mels, T] → SOT string. SSA runs two activity-conditioned passes."""
        if self.cfg.inject_mode == "ssa":
            if hard_mask is None:
                raise ValueError("SSA decode requires per-speaker activity (hard_mask)")
            texts = []
            mel_b = mel.unsqueeze(0)
            for k in (0, 1):
                act = hard_mask[:, k].unsqueeze(0).unsqueeze(-1)
                feats, _ = self.encode(mel_b, hard_mask=act)
                texts.append(self.greedy_decode(feats, max_tokens))
            return f"<spk0> {texts[0]} <spk1> {texts[1]}"
        feats, _ = self.encode(
            mel.unsqueeze(0),
            posterior=None if posterior is None else posterior.unsqueeze(0),
            hard_mask=None if hard_mask is None else hard_mask.unsqueeze(0),
        )
        return self.greedy_decode(feats, max_tokens)

    def forward(
        self,
        mel: torch.Tensor,
        tokens: torch.Tensor,
        posterior: Optional[torch.Tensor] = None,
        hard_mask: Optional[torch.Tensor] = None,
        overlap_mask: Optional[torch.Tensor] = None,
        spk_texts: Optional[list[tuple[str, str]]] = None,
        activity: Optional[torch.Tensor] = None,
    ) -> dict:
        audio_features, pred_logits = self.encode(
            mel, posterior, hard_mask, activity=activity
        )
        logits = self.whisper.decoder(tokens[:, :-1], audio_features)
        asr_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tokens[:, 1:].reshape(-1),
            ignore_index=-100,
        )

        diar_loss = torch.tensor(0.0, device=mel.device)
        if self.cfg.diar_loss_weight > 0:
            if self.cfg.diar_target == "hard" and hard_mask is not None:
                tgt_h = _align_time(hard_mask, pred_logits.size(1))
                dom = tgt_h.argmax(dim=-1)
                onehot = torch.zeros_like(pred_logits)
                onehot.scatter_(-1, dom.unsqueeze(-1), 1.0)
                silent = tgt_h.sum(dim=-1, keepdim=True) <= 0
                tgt = torch.where(silent, torch.zeros_like(onehot), onehot)
            elif posterior is not None:
                tgt = _align_time(posterior, pred_logits.size(1))
            else:
                tgt = None
            if tgt is not None:
                log_p = F.log_softmax(pred_logits, dim=-1)
                per = -(tgt * log_p).sum(-1)
                active = tgt.sum(dim=-1) > 0
                if overlap_mask is not None:
                    om = overlap_mask
                    if om.size(1) != per.size(1):
                        om = F.interpolate(
                            om.unsqueeze(1).float(),
                            size=per.size(1),
                            mode="nearest",
                        ).squeeze(1)
                    w = 1.0 + (self.cfg.overlap_loss_weight - 1.0) * om
                    w = w * active.float()
                    diar_loss = (per * w).sum() / w.sum().clamp_min(1.0)
                else:
                    diar_loss = (per * active.float()).sum() / active.float().sum().clamp_min(
                        1.0
                    )

        ctc_loss = torch.tensor(0.0, device=mel.device)
        if (
            self.cfg.aux_sdctc_weight > 0
            and self.ctc_head is not None
            and spk_texts is not None
        ):
            enc_for_ctc = audio_features
            k = self.cfg.decoder_prompt_tokens if self.decoder_prompt_mod is not None else 0
            if k > 0 and enc_for_ctc.size(1) > k:
                enc_for_ctc = enc_for_ctc[:, k:, :]
            tok_logits = self.ctc_head(enc_for_ctc)
            ctc_loss = sdctc_loss(pred_logits, tok_logits, spk_texts)

        total = (
            asr_loss
            + self.cfg.diar_loss_weight * diar_loss
            + self.cfg.aux_sdctc_weight * ctc_loss
        )
        return {
            "loss": total,
            "asr_loss": asr_loss.detach(),
            "diar_loss": diar_loss.detach() if torch.is_tensor(diar_loss) else diar_loss,
            "ctc_loss": ctc_loss.detach() if torch.is_tensor(ctc_loss) else ctc_loss,
            "logits": logits,
            "pred_logits": pred_logits,
        }
