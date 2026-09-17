"""cpWER / ORC-WER for 2-speaker SOT outputs.

cpWER follows CHiME-6 / MeetEval concatenated minimum-permutation WER:
assign hypothesis speakers to references with a permutation, sum
Levenshtein errors over the two streams, and divide by the total
number of reference words. The minimum over permutations is reported.

ORC-WER follows MeetEval: assign each reference utterance to a hypothesis
stream (speaker labels ignored), sum stream Levenshtein costs, and divide
by the number of reference words.
"""

from __future__ import annotations

import re
from itertools import permutations



_SPK_SPLIT = re.compile(r"<\s*spk\s*(\d+)\s*>", re.IGNORECASE)


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9'\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def split_sot(text: str, n_spk: int = 2) -> list[str]:
    parts = _SPK_SPLIT.split(text)
    by_id: dict[int, str] = {i: "" for i in range(n_spk)}
    if len(parts) == 1:
        by_id[0] = normalize_text(parts[0])
        return [by_id[i] for i in range(n_spk)]
    i = 1
    while i + 1 < len(parts):
        spk = int(parts[i])
        frag = normalize_text(parts[i + 1])
        if spk in by_id:
            by_id[spk] = (by_id[spk] + " " + frag).strip()
        i += 2
    return [by_id[i] for i in range(n_spk)]


def _levenshtein(ref_w: list[str], hyp_w: list[str]) -> int:
    """Word-level Levenshtein distance (S + D + I)."""
    n, m = len(ref_w), len(hyp_w)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i, rw in enumerate(ref_w, 1):
        cur = [i]
        for j, hw in enumerate(hyp_w, 1):
            cur.append(
                min(
                    cur[j - 1] + 1,
                    prev[j] + 1,
                    prev[j - 1] + (rw != hw),
                )
            )
        prev = cur
    return prev[-1]


def _edit_counts(ref: str, hyp: str) -> tuple[int, int]:
    """Return (S + D + I, number of reference words)."""
    ref_w = ref.split()
    hyp_w = hyp.split()
    return _levenshtein(ref_w, hyp_w), len(ref_w)


def cp_wer_counts(hyp_sot: str, ref0: str, ref1: str) -> tuple[int, int]:
    """MeetEval cpWER counts: (min-permutation errors, reference words)."""
    hyps = split_sot(hyp_sot, 2)
    refs = [normalize_text(ref0), normalize_text(ref1)]
    n_ref = sum(len(r.split()) for r in refs)
    if n_ref == 0:
        return 0, 0
    best_err = None
    for perm in permutations(range(2)):
        err = 0
        for i, r in enumerate(refs):
            e, _ = _edit_counts(r, hyps[perm[i]])
            err += e
        if best_err is None or err < best_err:
            best_err = err
    return int(best_err), n_ref


def cp_wer(hyp_sot: str, ref0: str, ref1: str) -> float:
    """Concatenated minimum-permutation WER (CHiME-6 / MeetEval)."""
    err, n = cp_wer_counts(hyp_sot, ref0, ref1)
    if n == 0:
        return 0.0
    return float(err / n)


def orc_wer_counts(hyp_sot: str, ref0: str, ref1: str) -> tuple[int, int]:
    """MeetEval ORC-WER counts: (min assignment errors, reference words).

    Each reference utterance is assigned to one hypothesis stream (speaker
    labels ignored). Utterances on the same stream are concatenated in
    reference order. The two-stream Levenshtein costs are summed.
    """
    hyps = split_sot(hyp_sot, 2)
    refs = [normalize_text(ref0), normalize_text(ref1)]
    n_ref = sum(len(r.split()) for r in refs)
    if n_ref == 0:
        return 0, 0
    best_err = None
    for a0 in (0, 1):
        for a1 in (0, 1):
            grouped = ["", ""]
            for r, a in zip(refs, (a0, a1)):
                if r:
                    grouped[a] = (grouped[a] + " " + r).strip()
            err = 0
            for j in range(2):
                e, _ = _edit_counts(grouped[j], hyps[j])
                err += e
            if best_err is None or err < best_err:
                best_err = err
    return int(best_err), n_ref


def orc_wer(hyp_sot: str, ref0: str, ref1: str) -> float:
    """Optimal reference combination WER (MeetEval)."""
    err, n = orc_wer_counts(hyp_sot, ref0, ref1)
    if n == 0:
        return 0.0
    return float(err / n)
