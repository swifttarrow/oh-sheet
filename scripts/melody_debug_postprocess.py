#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import mido


@dataclass
class MelodyNote:
    start: float
    duration: float
    pitch: int
    velocity: int

    @property
    def end(self) -> float:
        return self.start + self.duration


def load_monophonic_melody(path: Path) -> list[MelodyNote]:
    midi = mido.MidiFile(str(path))
    active: dict[int, list[tuple[float, int]]] = {}
    notes: list[MelodyNote] = []
    current_time = 0.0
    for msg in midi:
        current_time += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            active.setdefault(msg.note, []).append((current_time, msg.velocity))
            continue
        if msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            stack = active.get(msg.note)
            if not stack:
                continue
            start, velocity = stack.pop(0)
            notes.append(
                MelodyNote(
                    start=start,
                    duration=current_time - start,
                    pitch=msg.note,
                    velocity=velocity,
                )
            )
    notes.sort(key=lambda note: (note.start, note.pitch, note.duration))
    return notes


def repair_phrase_register(notes: list[MelodyNote]) -> list[MelodyNote]:
    repaired = [
        MelodyNote(
            start=note.start,
            duration=note.duration,
            pitch=note.pitch,
            velocity=note.velocity,
        )
        for note in notes
    ]
    segments: list[list[int]] = []
    current = [0]
    for index, (previous, candidate) in enumerate(zip(repaired, repaired[1:]), start=1):
        if candidate.start - previous.end <= 1.0:
            current.append(index)
        else:
            segments.append(current)
            current = [index]
    if current:
        segments.append(current)

    for segment in segments:
        if len(segment) >= 3:
            for offset in range(1, len(segment) - 1):
                prev_note = repaired[segment[offset - 1]]
                cur_note = repaired[segment[offset]]
                next_note = repaired[segment[offset + 1]]
                original_cost = abs(cur_note.pitch - prev_note.pitch) + abs(
                    next_note.pitch - cur_note.pitch
                )

                if (
                    cur_note.pitch > prev_note.pitch + 9
                    and cur_note.pitch > next_note.pitch + 9
                    and 67 <= cur_note.pitch - 12 <= 84
                ):
                    candidate_pitch = cur_note.pitch - 12
                    candidate_cost = abs(candidate_pitch - prev_note.pitch) + abs(
                        next_note.pitch - candidate_pitch
                    )
                    if candidate_cost <= original_cost - 8:
                        cur_note.pitch = candidate_pitch
                        continue

                if (
                    cur_note.pitch < prev_note.pitch - 9
                    and cur_note.pitch < next_note.pitch - 9
                    and 67 <= cur_note.pitch + 12 <= 84
                ):
                    candidate_pitch = cur_note.pitch + 12
                    candidate_cost = abs(candidate_pitch - prev_note.pitch) + abs(
                        next_note.pitch - candidate_pitch
                    )
                    if candidate_cost <= original_cost - 8:
                        cur_note.pitch = candidate_pitch

        if len(segment) >= 2:
            first = repaired[segment[0]]
            second = repaired[segment[1]]
            if first.pitch > second.pitch + 10 and 67 <= first.pitch - 12 <= 84:
                candidate_pitch = first.pitch - 12
                if abs(candidate_pitch - second.pitch) <= abs(first.pitch - second.pitch) - 8:
                    first.pitch = candidate_pitch

        if len(segment) == 2:
            first = repaired[segment[0]]
            second = repaired[segment[1]]
            if (
                second.duration >= 4.0
                and second.pitch - first.pitch >= 12
                and 67 <= first.pitch + 12 <= 84
            ):
                candidate_pitch = first.pitch + 12
                if abs(second.pitch - candidate_pitch) <= abs(second.pitch - first.pitch) - 8:
                    first.pitch = candidate_pitch

    return repaired


def lift_same_pitch_anchor_drops(notes: list[MelodyNote]) -> list[MelodyNote]:
    repaired = [
        MelodyNote(
            start=note.start,
            duration=note.duration,
            pitch=note.pitch,
            velocity=note.velocity,
        )
        for note in notes
    ]
    for index in range(1, len(repaired)):
        previous = repaired[index - 1]
        current = repaired[index]
        following = repaired[index + 1] if index + 1 < len(repaired) else None
        if following is None:
            continue
        gap = current.start - previous.end
        if (
            gap <= 0.08
            and previous.pitch - current.pitch == 12
            and previous.duration >= 1.5
            and current.duration >= 0.8
            and following.pitch >= current.pitch + 5
            and following.start - current.end <= 2.5
            and current.pitch + 12 <= 84
        ):
            current.pitch += 12
    return repaired


def collapse_same_pitch_class_octave_toggles(notes: list[MelodyNote]) -> list[MelodyNote]:
    repaired = [
        MelodyNote(
            start=note.start,
            duration=note.duration,
            pitch=note.pitch,
            velocity=note.velocity,
        )
        for note in notes
    ]
    for index in range(1, len(repaired) - 1):
        previous = repaired[index - 1]
        current = repaired[index]
        following = repaired[index + 1]

        if (
            previous.pitch % 12 == current.pitch % 12 == following.pitch % 12
            and abs(previous.pitch - current.pitch) == 12
            and abs(following.pitch - current.pitch) == 12
            and previous.pitch == following.pitch
        ):
            current.pitch = previous.pitch
            continue

        gap_to_next = following.start - current.end
        if (
            current.pitch % 12 == following.pitch % 12
            and abs(current.pitch - following.pitch) == 12
            and current.duration <= 0.7
            and gap_to_next <= 0.7
            and abs(previous.pitch - following.pitch) <= 7
        ):
            current.pitch = following.pitch

    return repaired


def merge_same_pitch_continuations(notes: list[MelodyNote]) -> list[MelodyNote]:
    merged: list[MelodyNote] = []
    for note in notes:
        if merged:
            previous = merged[-1]
            if previous.pitch == note.pitch and 0 <= note.start - previous.end <= 0.08:
                previous.duration = (note.start + note.duration) - previous.start
                continue
        merged.append(
            MelodyNote(
                start=note.start,
                duration=note.duration,
                pitch=note.pitch,
                velocity=note.velocity,
            )
        )
    return merged


def write_melody(path: Path, notes: list[MelodyNote], bpm: float) -> None:
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=int(round(60_000_000 / bpm)), time=0))
    track.append(mido.Message("program_change", program=0, time=0))
    seconds_per_beat = 60.0 / bpm
    events: list[tuple[int, bool, int, int]] = []
    for note in notes:
        events.append((int(round(note.start / seconds_per_beat * 480)), True, note.pitch, note.velocity))
        events.append((int(round(note.end / seconds_per_beat * 480)), False, note.pitch, 0))
    events.sort(key=lambda event: (event[0], not event[1], event[2]))
    last_tick = 0
    for tick, is_note_on, pitch, velocity in events:
        delta = max(0, tick - last_tick)
        last_tick = tick
        track.append(
            mido.Message(
                "note_on" if is_note_on else "note_off",
                note=pitch,
                velocity=velocity,
                time=delta,
            )
        )
    midi.save(path)


def summarize(notes: list[MelodyNote]) -> str:
    if len(notes) < 2:
        return f"melody={len(notes)}"
    avg_duration = sum(note.duration for note in notes) / len(notes)
    short_share = sum(note.duration <= 0.5 for note in notes) / len(notes)
    large_leaps = sum(abs(curr.pitch - prev.pitch) >= 9 for prev, curr in zip(notes, notes[1:]))
    mean_interval = sum(abs(curr.pitch - prev.pitch) for prev, curr in zip(notes, notes[1:])) / (
        len(notes) - 1
    )
    return (
        f"melody={len(notes)} avg_dur={avg_duration:.3f} short_share={short_share:.3f} "
        f"large_leaps={large_leaps} mean_interval={mean_interval:.3f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Post-process a melody-only MIDI with phrase register repair and anchor cleanup."
    )
    parser.add_argument("input_midi", type=Path)
    parser.add_argument("-o", "--output-midi", type=Path, required=True)
    parser.add_argument("--bpm", type=float, default=108.0)
    args = parser.parse_args()

    notes = load_monophonic_melody(args.input_midi)
    notes = repair_phrase_register(notes)
    notes = lift_same_pitch_anchor_drops(notes)
    notes = collapse_same_pitch_class_octave_toggles(notes)
    notes = merge_same_pitch_continuations(notes)
    write_melody(args.output_midi, notes, args.bpm)
    print(f"Wrote melody MIDI: {args.output_midi}")
    print(summarize(notes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
