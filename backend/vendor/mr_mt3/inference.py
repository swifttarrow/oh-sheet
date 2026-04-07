"""Pretrained MT3 inference — slimmed from upstream ``transcribe_mrmt3.py``.

Audio file → ``note_seq.NoteSequence``. The conversion to our pydantic
``TranscriptionResult`` lives in ``backend.services.transcribe`` so this
module stays free of any oh-sheet imports — it only depends on its
sibling modules under ``backend.vendor.mr_mt3``.

Stripped from the upstream script:
  * pitch-shift ensemble (``transcribe_ensemble``)
  * CQT spectral hallucination filter
  * the ``ns_to_transcription_result`` adapter that depended on a
    different contracts module
  * the CLI ``main()``

Kept (used by the transcribe service):
  * ``load_model``
  * ``transcribe``
  * ``rescale_velocity_to_rms``
"""
from __future__ import annotations

import json
import math
import os

import librosa
import numpy as np
import torch

from .contrib import metrics_utils, note_sequences, spectrograms, vocabularies
from .models.t5 import T5Config, T5ForConditionalGeneration

_HERE = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_HERE, "config", "mt3_config.json")
DEFAULT_CHECKPOINT_PATH = os.path.join(_HERE, "pretrained", "mt3.pth")


def load_model(checkpoint_path: str, device: torch.device):
    """Load the pretrained MT3 raw state_dict from disk."""
    with open(_CONFIG_PATH) as f:
        config_dict = json.load(f)
    config = T5Config.from_dict(config_dict)
    model = T5ForConditionalGeneration(config)

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=True)

    model.eval()
    model.to(device)
    return model, config


def load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    """Load audio file as 16kHz mono float32."""
    audio, _ = librosa.load(path, sr=target_sr, mono=True)
    return audio


def audio_to_frames(audio: np.ndarray, spectrogram_config):
    """Compute spectrogram frames from audio."""
    frame_size = spectrogram_config.hop_width
    padding = [0, frame_size - len(audio) % frame_size]
    audio = np.pad(audio, padding, mode="constant")
    frames = spectrograms.split_audio(audio, spectrogram_config)
    num_frames = len(audio) // frame_size
    times = np.arange(num_frames) / spectrogram_config.frames_per_second
    return frames, times


def split_into_segments(frames, frame_times, max_length: int = 256):
    """Split frames into fixed-length segments for the model."""
    num_segments = math.ceil(frames.shape[0] / max_length)
    segments = []
    times_list = []
    paddings = []
    for i in range(num_segments):
        seg = np.zeros((max_length, *frames.shape[1:]))
        t = np.zeros(max_length)
        start = i * max_length
        end = min(start + max_length, frames.shape[0])
        length = end - start
        seg[:length] = frames[start:end]
        t[:length] = frame_times[start:end]
        segments.append(seg)
        times_list.append(t)
        paddings.append(length)
    return np.stack(segments), np.stack(times_list), paddings


def compute_spectrograms(inputs, spectrogram_config):
    """Compute mel spectrograms from frame segments."""
    outputs = []
    for seg in inputs:
        samples = spectrograms.flatten_frames(seg, spectrogram_config.use_tf_spectral_ops)
        mel = spectrograms.compute_spectrogram(samples, spectrogram_config)
        outputs.append(mel)
    return np.stack(outputs)


def tokens_to_midi(predictions, frame_times, codec, vocab):
    """Convert model output tokens to a NoteSequence."""
    all_predictions = []
    for i, batch in enumerate(predictions):
        for j, tokens in enumerate(batch):
            eos_mask = tokens == vocabularies.DECODED_EOS_ID
            if eos_mask.any():
                tokens = tokens[: np.argmax(eos_mask)]
            start_time = frame_times[i][j][0]
            start_time -= start_time % (1 / codec.steps_per_second)
            all_predictions.append({
                "est_tokens": tokens,
                "start_time": start_time,
                "raw_inputs": [],
            })

    encoding_spec = note_sequences.NoteEncodingWithTiesSpec
    result = metrics_utils.event_predictions_to_ns(
        all_predictions, codec=codec, encoding_spec=encoding_spec
    )
    return result["est_ns"]


@torch.no_grad()
def transcribe(
    audio_path: str,
    model,
    device: torch.device,
    *,
    batch_size: int = 4,
    max_length: int = 1024,
):
    """Full pipeline: audio file → NoteSequence.

    The codec is hardcoded to ``num_velocity_bins=1`` to match the
    pretrained mt3.pth checkpoint (verified empirically: vb=1 decodes the
    model's emitted tokens with 0 invalid events; any other value produces
    invalid events and corrupts program/drum decoding).
    """
    spec_config = spectrograms.SpectrogramConfig()
    spec_config.use_tf_spectral_ops = False
    codec = vocabularies.build_codec(
        vocab_config=vocabularies.VocabularyConfig(num_velocity_bins=1)
    )
    vocab = vocabularies.vocabulary_from_codec(codec)

    audio = load_audio(audio_path)
    frames, frame_times = audio_to_frames(audio, spec_config)
    segments, seg_times, paddings = split_into_segments(frames, frame_times)
    inputs = compute_spectrograms(segments, spec_config)

    # Zero out padding
    for i, p in enumerate(paddings):
        inputs[i, p:] = 0

    inputs_tensor = torch.from_numpy(inputs).float()
    all_results = []
    all_times = []

    for start in range(0, len(inputs_tensor), batch_size):
        end = min(start + batch_size, len(inputs_tensor))
        batch = inputs_tensor[start:end].to(device)
        result = model.generate(inputs=batch, max_length=max_length)
        # Postprocess: mask after EOS, subtract special tokens
        after_eos = torch.cumsum((result == model.config.eos_token_id).float(), dim=-1)
        result = result - vocab.num_special_tokens()
        result = torch.where(after_eos.bool(), -1, result)
        result = result[:, 1:]  # remove BOS
        all_results.append(result.cpu().numpy())
        all_times.append(seg_times[start:end])

    return tokens_to_midi(all_results, all_times, codec, vocab)


def rescale_velocity_to_rms(ns, audio_path: str, sr: int = 22050, hop: int = 512):
    """Rescale note velocities using onset RMS from the source audio.

    MR-MT3 trained with num_velocity_bins=1 outputs all velocity=127. This
    maps each note's onset energy to velocity 40–127 so dynamics survive
    into the score.
    """
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    rms = librosa.feature.rms(y=audio, hop_length=hop)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    rms_max = float(np.percentile(rms, 95)) + 1e-10  # robust max

    for note in ns.notes:
        frame = np.searchsorted(rms_times, note.start_time)
        end_frame = min(frame + max(1, int(0.05 * sr / hop)), len(rms))
        if frame >= len(rms):
            note.velocity = 80  # fallback
            continue
        onset_rms = float(np.mean(rms[frame:end_frame]))
        note.velocity = int(np.clip(40 + (onset_rms / rms_max) * 87, 40, 127))

    return ns
