"""Tests for per-pitch CQT-based duration refinement."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.services.duration_refine import (
    _CQT_FMIN_MIDI,
    refine_durations,
)
from backend.services.transcription_cleanup import NoteEvent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cqt(n_bins: int, n_frames: int, pitch_midi: int, decay_frame: int):
    C = np.zeros((n_bins, n_frames), dtype=np.float32)
    cqt_bin = pitch_midi - _CQT_FMIN_MIDI
    if 0 <= cqt_bin < n_bins:
        C[cqt_bin, :decay_frame] = 1.0
        C[cqt_bin, decay_frame:] = 0.01
    return C


def _make_mock_librosa(fake_audio, sr, fake_cqt):
    mock = MagicMock()
    mock.load.return_value = (fake_audio, sr)
    mock.cqt.return_value = fake_cqt
    return mock


SR = 22050
HOP = 256
N_BINS = 84
FRAME_DUR = HOP / SR


@pytest.fixture(autouse=True)
def _clear_librosa_cache():
    sys.modules.pop("librosa", None)
    yield
    sys.modules.pop("librosa", None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRefineShortensDecayedNote:
    def test_note_trimmed_at_decay(self):
        pitch = 60
        note_start = 0.0
        note_end = 1.0
        decay_frame = 40
        n_frames = int(note_end / FRAME_DUR) + 10

        fake_cqt = _make_cqt(N_BINS, n_frames, pitch, decay_frame)
        fake_audio = np.zeros(int(SR * note_end), dtype=np.float32)
        mock_lr = _make_mock_librosa(fake_audio, SR, fake_cqt)

        events: list[NoteEvent] = [(note_start, note_end, pitch, 0.8, None)]

        with patch.dict(sys.modules, {"librosa": mock_lr}):
            refined, stats = refine_durations(
                events, Path("/fake/audio.wav"), sr=SR, hop_length=HOP
            )

        assert len(refined) == 1
        assert refined[0][1] < note_end
        assert stats.refined_count == 1
        assert stats.mean_trim_sec > 0
        assert refined[0][0] == note_start


class TestMinDurationRespected:
    def test_min_duration_floor(self):
        pitch = 60
        note_start = 0.5
        note_end = 1.5
        n_frames = 200
        fake_cqt = _make_cqt(N_BINS, n_frames, pitch, 1)
        fake_audio = np.zeros(SR * 2, dtype=np.float32)
        mock_lr = _make_mock_librosa(fake_audio, SR, fake_cqt)

        events: list[NoteEvent] = [(note_start, note_end, pitch, 0.8, None)]
        min_dur = 0.05

        with patch.dict(sys.modules, {"librosa": mock_lr}):
            refined, stats = refine_durations(
                events, Path("/fake/audio.wav"), sr=SR, hop_length=HOP,
                min_duration_sec=min_dur,
            )

        assert len(refined) == 1
        actual_dur = refined[0][1] - refined[0][0]
        assert actual_dur >= min_dur


class TestNeverExtended:
    def test_note_not_extended(self):
        pitch = 60
        note_start = 0.0
        note_end = 0.5
        n_frames = 200
        fake_cqt = np.ones((N_BINS, n_frames), dtype=np.float32)
        fake_audio = np.zeros(SR * 3, dtype=np.float32)
        mock_lr = _make_mock_librosa(fake_audio, SR, fake_cqt)

        events: list[NoteEvent] = [(note_start, note_end, pitch, 0.8, None)]

        with patch.dict(sys.modules, {"librosa": mock_lr}):
            refined, stats = refine_durations(
                events, Path("/fake/audio.wav"), sr=SR, hop_length=HOP
            )

        assert len(refined) == 1
        assert refined[0][1] <= note_end
        assert stats.refined_count == 0


class TestGracefulFallback:
    def test_audio_load_failure(self):
        events: list[NoteEvent] = [(0.0, 1.0, 60, 0.8, None)]
        mock_lr = MagicMock()
        mock_lr.load.side_effect = RuntimeError("file not found")

        with patch.dict(sys.modules, {"librosa": mock_lr}):
            refined, stats = refine_durations(events, Path("/nonexistent.wav"))

        assert refined == list(events)
        assert stats.refined_count == 0


class TestEmptyEvents:
    def test_empty_list(self):
        refined, stats = refine_durations([], Path("/fake.wav"))
        assert refined == []
        assert stats.total_notes == 0


class TestPitchOutsideCqtRange:
    def test_very_low_pitch(self):
        events: list[NoteEvent] = [(0.0, 1.0, 20, 0.5, None)]
        n_frames = 200
        fake_cqt = np.ones((N_BINS, n_frames), dtype=np.float32)
        fake_audio = np.zeros(SR * 2, dtype=np.float32)
        mock_lr = _make_mock_librosa(fake_audio, SR, fake_cqt)

        with patch.dict(sys.modules, {"librosa": mock_lr}):
            refined, stats = refine_durations(
                events, Path("/fake.wav"), sr=SR, hop_length=HOP
            )

        assert refined[0][1] == 1.0
        assert stats.refined_count == 0

    def test_very_high_pitch(self):
        events: list[NoteEvent] = [(0.0, 1.0, 110, 0.5, None)]
        n_frames = 200
        fake_cqt = np.ones((N_BINS, n_frames), dtype=np.float32)
        fake_audio = np.zeros(SR * 2, dtype=np.float32)
        mock_lr = _make_mock_librosa(fake_audio, SR, fake_cqt)

        with patch.dict(sys.modules, {"librosa": mock_lr}):
            refined, stats = refine_durations(
                events, Path("/fake.wav"), sr=SR, hop_length=HOP
            )

        assert refined[0][1] == 1.0
        assert stats.refined_count == 0
