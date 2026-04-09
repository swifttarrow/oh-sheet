#!/usr/bin/env python3
"""
Merge every track in a MIDI file into one track on channel 0 using GM Acoustic Grand Piano (program 0).

Requires: pip install mido
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import mido
except ImportError:
    print("This script needs the `mido` package. Install with: pip install mido", file=sys.stderr)
    sys.exit(1)

# General MIDI program 0 — Acoustic Grand Piano
GM_ACOUSTIC_GRAND = 0


def load_midi(in_path: Path) -> tuple[mido.MidiFile, bool]:
    """Load MIDI strictly, then fall back to clipping invalid data bytes."""
    try:
        return mido.MidiFile(in_path), False
    except OSError as exc:
        if "data byte must be in range 0..127" not in str(exc):
            raise

    # Some files contain stray bytes >127 inside channel messages.
    # mido can recover by clipping those bytes into the legal MIDI range.
    return mido.MidiFile(in_path, clip=True), True


def _event_sort_order(msg: mido.Message) -> int:
    """Order simultaneous events for broad player compatibility."""
    if msg.is_meta:
        if msg.type == "set_tempo":
            return 0
        if msg.type == "time_signature":
            return 1
        if msg.type == "key_signature":
            return 2
        if msg.type == "end_of_track":
            return 200
        return 10
    if msg.type == "program_change":
        return 20
    return 50


def merge_to_single_piano(mid: mido.MidiFile) -> mido.MidiFile:
    merged = mido.merge_tracks(mid.tracks)
    abs_t = 0
    events: list[tuple[int, mido.Message]] = []
    for msg in merged:
        abs_t += msg.time
        if msg.is_meta:
            events.append((abs_t, msg.copy()))
        elif msg.type == "program_change":
            continue
        else:
            m = msg.copy()
            m.channel = 0
            events.append((abs_t, m))

    events.append(
        (0, mido.Message("program_change", channel=0, program=GM_ACOUSTIC_GRAND, time=0)),
    )
    events.sort(key=lambda te: (te[0], _event_sort_order(te[1])))

    new_track = mido.MidiTrack()
    prev = 0
    for t, msg in events:
        msg = msg.copy()
        msg.time = t - prev
        new_track.append(msg)
        prev = t

    out = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)
    out.type = 0
    out.tracks = [new_track]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge all MIDI tracks into one piano track (GM Acoustic Grand, channel 0).",
    )
    parser.add_argument("input", type=Path, help="Input .mid / .midi file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path (default: <input_stem>_piano.mid next to input)",
    )
    args = parser.parse_args()
    in_path: Path = args.input

    if not in_path.is_file():
        print(f"Not a file: {in_path}", file=sys.stderr)
        return 1

    suffix = in_path.suffix.lower()
    if suffix not in (".mid", ".midi"):
        print("Warning: input does not end in .mid or .midi — continuing anyway.", file=sys.stderr)

    out_path = args.output
    if out_path is None:
        out_path = in_path.with_name(f"{in_path.stem}_piano.mid")

    try:
        mid, clipped_invalid_bytes = load_midi(in_path)
    except Exception as e:
        print(f"Failed to read MIDI: {e}", file=sys.stderr)
        return 1
    if clipped_invalid_bytes:
        print(
            "Warning: input MIDI contains invalid data bytes; clipped them into the 0..127 range.",
            file=sys.stderr,
        )

    out_mid = merge_to_single_piano(mid)
    try:
        out_mid.save(out_path)
    except Exception as e:
        print(f"Failed to write MIDI: {e}", file=sys.stderr)
        return 1

    print(f"Wrote {out_path} (1 track, {out_mid.ticks_per_beat} ticks/beat)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
