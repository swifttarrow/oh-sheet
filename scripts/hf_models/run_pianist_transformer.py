#!/usr/bin/env python3
"""
Pianist Transformer rendering model (yhj137/pianist-transformer-rendering).

**Golden input:** ``golden.mid`` (score MIDI → expressive performance). Not an audio model.

Uses the official GitHub repo for inference code (midi_to_ids, batch_performance_render)
and Hugging Face for weights. First run clones ~source and downloads ~272MB weights.

Install (prefer a fresh venv; versions mirror upstream README):
  pip install torch miditoolkit tqdm "transformers>=4.54" accelerate datasets

Recent ``transformers`` releases moved T5Gemma mask helpers out of
``modeling_t5gemma``; this script patches that module with the small factories
PianistTransformer still imports (same behavior as transformers v4.55).

``PianoT5Gemma`` still declares ``_tied_weights_keys`` as a **list**; current
``transformers`` expects a **dict** (same mapping as upstream
``T5GemmaForConditionalGeneration``). The script normalizes that on the class
before ``from_pretrained``.

CUDA optional; CPU works but is slower.

Weights are public — HF_TOKEN only if you hit rate limits or use private mirrors.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Optional

_HF_DIR = Path(__file__).resolve().parent
if str(_HF_DIR) not in sys.path:
    sys.path.insert(0, str(_HF_DIR))

import _common  # noqa: E402

_GIT_URL = "https://github.com/yhj137/PianistTransformer.git"
_HF_REPO = "yhj137/pianist-transformer-rendering"
_CACHE = Path.home() / ".cache" / "oh-sheet-hf" / "PianistTransformer"


def _patch_t5gemma_modeling_for_pianist() -> None:
    """Restore mask factories that Pianist's ``pianoformer.py`` imports from ``modeling_t5gemma``.

    Hugging Face refactored T5Gemma (helpers live in ``masking_utils``); upstream Pianist
    still expects the v4.55-style API on ``transformers.models.t5gemma.modeling_t5gemma``.
    """
    import torch

    import transformers.models.t5gemma.modeling_t5gemma as m

    if getattr(m, "_ohsheet_pianist_t5gemma_mask_patch", False):
        return

    def bidirectional_mask_function(attention_mask: Optional[torch.Tensor]) -> Callable[..., torch.Tensor | bool]:
        # Must return tensors (no ``.item()``): ``create_causal_mask`` composes masks under ``vmap``.
        def inner_mask(batch_idx, head_idx, q_idx, kv_idx):
            if attention_mask is None:
                return q_idx.new_tensor(True, dtype=torch.bool)
            return attention_mask[batch_idx, kv_idx].to(dtype=torch.bool)

        return inner_mask

    def sliding_window_bidirectional_mask_function(sliding_window: int) -> Callable[..., torch.Tensor | bool]:
        def inner_mask(batch_idx, head_idx, q_idx, kv_idx):
            return (q_idx - sliding_window < kv_idx) & (kv_idx < q_idx + sliding_window)

        return inner_mask

    m.bidirectional_mask_function = bidirectional_mask_function  # type: ignore[attr-defined]
    m.sliding_window_bidirectional_mask_function = sliding_window_bidirectional_mask_function  # type: ignore[attr-defined]
    m._ohsheet_pianist_t5gemma_mask_patch = True  # type: ignore[attr-defined]


def _patch_pianist_pianoformer_tied_weights() -> None:
    """``PianoT5Gemma._tied_weights_keys`` is a list in upstream; ``transformers`` requires a dict."""
    from src.model.pianoformer import PianoT5Gemma

    if getattr(PianoT5Gemma, "_ohsheet_tied_weights_patch", False):
        return

    tw = getattr(PianoT5Gemma, "_tied_weights_keys", None)
    if isinstance(tw, dict):
        PianoT5Gemma._ohsheet_tied_weights_patch = True  # type: ignore[attr-defined]
        return

    # Match ``T5GemmaForConditionalGeneration`` in current ``transformers``.
    PianoT5Gemma._tied_weights_keys = {  # type: ignore[misc]
        "lm_head.out_proj.weight": "model.decoder.embed_tokens.weight",
    }
    PianoT5Gemma._ohsheet_tied_weights_patch = True  # type: ignore[attr-defined]


def _ensure_git_clone() -> Path:
    if _CACHE.is_dir() and (_CACHE / "src").is_dir():
        return _CACHE
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", _GIT_URL, str(_CACHE)],
        check=True,
    )
    return _CACHE


def _ensure_weights(repo: Path) -> Path:
    sft = repo / "models" / "sft"
    if (sft / "model.safetensors").is_file() or (sft / "model.bin").is_file():
        return sft
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        print("pip install huggingface_hub", file=sys.stderr)
        raise e
    sft.mkdir(parents=True, exist_ok=True)
    snapshot_download(_HF_REPO, local_dir=str(sft))
    return sft


def main() -> int:
    p = _common.midi_arg()
    args = p.parse_args()

    midi_in = _common.require_midi(args.midi)
    out = args.out or (_common.repo_root() / "hf_model_outputs" / "pianist_rendered.mid")
    out = Path(out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    repo = _ensure_git_clone()
    sft_dir = _ensure_weights(repo)

    score_dir = repo / "data" / "midis" / "testset" / "score"
    inf_dir = repo / "data" / "midis" / "testset" / "inference"
    score_dir.mkdir(parents=True, exist_ok=True)
    inf_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(midi_in, score_dir / "0.mid")

    prev_cwd = Path.cwd()
    prev_path = list(sys.path)
    try:
        os.chdir(repo)
        sys.path.insert(0, str(repo))

        import torch
        from miditoolkit import MidiFile

        _patch_t5gemma_modeling_for_pianist()

        from src.model.generate import batch_performance_render, map_midi

        _patch_pianist_pianoformer_tied_weights()
        from src.model.pianoformer import PianoT5Gemma

        dtype = torch.bfloat16
        if not torch.cuda.is_available():
            dtype = torch.float32

        print(f"Loading model from {sft_dir} …")
        model = PianoT5Gemma.from_pretrained(str(sft_dir), torch_dtype=dtype)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        midis = [MidiFile(str(score_dir / "0.mid"))]
        print("Rendering performance …")
        res = batch_performance_render(
            model,
            midis,
            temperature=1.0,
            top_p=0.95,
            device=device,
        )
        for i, mid in enumerate(res):
            mid = map_midi(midis[i], mid)
            mid.dump(str(out))
        print(f"Wrote {out}")
        return 0
    finally:
        os.chdir(prev_cwd)
        sys.path[:] = prev_path


if __name__ == "__main__":
    raise SystemExit(main())
