"""HF arrange path: materialize MIDI, inference stub, parse → PianoScore."""
from __future__ import annotations

import io

import pytest

from backend.config import settings
from backend.contracts import (
    SCHEMA_VERSION,
    HarmonicAnalysis,
    InputBundle,
    InputMetadata,
    InstrumentRole,
    MidiTrack,
    Note,
    QualitySignal,
    RemoteMidiFile,
    TempoMapEntry,
    TranscriptionResult,
)
from backend.jobs.runner import _bundle_to_transcription
from backend.services.arrange import ArrangeService, _arrange_hf_sync
from backend.services.hf_arrange import inference as hf_inference
from backend.services.transcription_midi_materialize import (
    materialize_transcription_midi_bytes,
    serialize_transcription_to_midi_bytes,
)
from backend.storage.local import LocalBlobStore


def _minimal_txr() -> TranscriptionResult:
    return TranscriptionResult(
        schema_version=SCHEMA_VERSION,
        midi_tracks=[
            MidiTrack(
                notes=[
                    Note(pitch=60, onset_sec=0.0, offset_sec=0.5, velocity=80),
                ],
                instrument=InstrumentRole.PIANO,
                program=0,
                confidence=0.9,
            ),
        ],
        analysis=HarmonicAnalysis(
            key="C:major",
            time_signature=(4, 4),
            tempo_map=[TempoMapEntry(time_sec=0.0, beat=0.0, bpm=120.0)],
            chords=[],
            sections=[],
        ),
        quality=QualitySignal(overall_confidence=0.9, warnings=[]),
    )


def test_serialize_transcription_round_trip_bytes() -> None:
    pytest.importorskip("pretty_midi")
    txr = _minimal_txr()
    raw = serialize_transcription_to_midi_bytes(txr)
    assert raw is not None and len(raw) > 32


def test_materialize_prefers_blob_uri(tmp_path) -> None:
    pytest.importorskip("pretty_midi")
    blob = LocalBlobStore(tmp_path)
    txr = _minimal_txr()
    uri = blob.put_bytes("jobs/t1/transcription/x.mid", b"not-valid-midi")
    txr = txr.model_copy(update={"transcription_midi_uri": uri})
    assert materialize_transcription_midi_bytes(txr, blob) == b"not-valid-midi"


def test_materialize_serializes_without_uri(tmp_path) -> None:
    pytest.importorskip("pretty_midi")
    blob = LocalBlobStore(tmp_path)
    txr = _minimal_txr()
    b = materialize_transcription_midi_bytes(txr, blob)
    assert b.startswith(b"MThd")


@pytest.mark.asyncio
async def test_hf_midi_identity_arrange(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    pytest.importorskip("pretty_midi")
    monkeypatch.setattr(settings, "arrange_backend", "hf_midi_identity")
    monkeypatch.setattr(settings, "arrange_hf_fallback_to_rules", False)

    blob = LocalBlobStore(tmp_path)
    txr = _minimal_txr()
    uri = blob.put_bytes(
        "jobs/hf-test/transcription/in.mid",
        serialize_transcription_to_midi_bytes(txr) or b"",
    )
    txr = txr.model_copy(update={"transcription_midi_uri": uri})

    svc = ArrangeService()
    score = await svc.run(txr, blob_store=blob)
    assert len(score.right_hand) + len(score.left_hand) >= 1


def test_bundle_to_transcription_sets_midi_uri(tmp_path) -> None:
    pytest.importorskip("pretty_midi")
    import pretty_midi  # noqa: PLC0415

    mid_path = tmp_path / "in.mid"
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(
        pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=0.5),
    )
    pm.instruments.append(inst)
    pm.write(str(mid_path))

    bundle = InputBundle(
        midi=RemoteMidiFile(uri=mid_path.as_uri(), ticks_per_beat=480),
        metadata=InputMetadata(title="t", artist="a", source="midi_upload"),
    )
    blob = LocalBlobStore(tmp_path / "blob")
    txr = _bundle_to_transcription(bundle, blob_store=blob, job_id="jid-1")
    assert txr.transcription_midi_uri is not None
    assert "upload.mid" in txr.transcription_midi_uri
    assert blob.get_bytes(txr.transcription_midi_uri) == mid_path.read_bytes()


@pytest.mark.asyncio
async def test_arrange_hf_fallback_on_bad_hf_output(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    pytest.importorskip("pretty_midi")
    monkeypatch.setattr(settings, "arrange_backend", "hf_midi_identity")
    monkeypatch.setattr(settings, "arrange_hf_fallback_to_rules", True)

    blob = LocalBlobStore(tmp_path)
    txr = _minimal_txr()
    uri = blob.put_bytes(
        "jobs/fb/transcription/in.mid",
        serialize_transcription_to_midi_bytes(txr) or b"",
    )
    txr = txr.model_copy(update={"transcription_midi_uri": uri})

    def _bad(_midi_in: bytes, _mode: str) -> bytes:
        return b"BAD"

    monkeypatch.setattr(hf_inference, "run_hf_midi_inference", _bad)

    svc = ArrangeService()
    score = await svc.run(txr, blob_store=blob)
    assert len(score.right_hand) + len(score.left_hand) >= 1


def test_arrange_hf_sync_monkeypatch_custom_midi(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    pytest.importorskip("pretty_midi")
    import pretty_midi  # noqa: PLC0415

    blob = LocalBlobStore(tmp_path)
    base = _minimal_txr()
    uri = blob.put_bytes(
        "jobs/sync/transcription/in.mid",
        serialize_transcription_to_midi_bytes(base) or b"",
    )
    base = base.model_copy(update={"transcription_midi_uri": uri})

    out_pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=0)
    for pitch, start in [(60, 0.0), (64, 0.5), (67, 1.0)]:
        inst.notes.append(
            pretty_midi.Note(velocity=90, pitch=pitch, start=start, end=start + 0.4),
        )
    out_pm.instruments.append(inst)
    buf = io.BytesIO()
    out_pm.write(buf)

    def _three_notes(midi_in: bytes, mode: str) -> bytes:  # noqa: ARG001
        return buf.getvalue()

    monkeypatch.setattr(hf_inference, "run_hf_midi_inference", _three_notes)
    score = _arrange_hf_sync(base, "intermediate", blob)
    n = len(score.right_hand) + len(score.left_hand)
    # Arrange quantizes and may merge close notes; just verify output is non-empty.
    assert n >= 1
