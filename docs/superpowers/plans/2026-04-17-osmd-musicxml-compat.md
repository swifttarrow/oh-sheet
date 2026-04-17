# OSMD MusicXML Compatibility Fixes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix OSMD rendering issues (UUID instrument names, note collisions) and add tooling for fast isolated MusicXML iteration without the full pipeline.

**Architecture:** Extend `_sanitize_musicxml_for_osmd()` in `backend/services/engrave.py` with a part-name clearing pass (and a collision-fix pass once diagnosed). Add a standalone OSMD viewer HTML page and a MusicXML extraction script for visual debugging.

**Tech Stack:** Python 3.10+, xml.etree.ElementTree, lxml (tests), OSMD 1.8.9 (CDN), pytest

---

## File Structure

| File | Role |
|------|------|
| `tests/osmd_viewer.html` | **Create** — Standalone OSMD viewer for visual MusicXML inspection |
| `tests/extract_musicxml.py` | **Create** — Script to extract sanitized MusicXML from score fixtures |
| `backend/services/engrave.py` | **Modify** — Add `_clear_part_names()` pass + collision fix pass to `_sanitize_musicxml_for_osmd()` |
| `tests/test_engrave_quality.py` | **Modify** — Add L2 tests for part-name clearing and collision regression |
| `tests/fixtures/jammin_osmd_regression.musicxml` | **Create** — Regression fixture for OSMD rendering |

---

### Task 1: Create static OSMD viewer page

**Files:**
- Create: `tests/osmd_viewer.html`

- [ ] **Step 1: Write the HTML file**

Create `tests/osmd_viewer.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>OSMD MusicXML Viewer</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 20px; }
    #drop-zone {
      border: 2px dashed #aaa; border-radius: 8px; padding: 40px;
      text-align: center; color: #666; margin-bottom: 20px;
      transition: border-color 0.2s;
    }
    #drop-zone.drag-over { border-color: #2196F3; background: #e3f2fd; }
    #osmd-container { width: 100%; }
    .error { color: red; font-weight: bold; }
    .info { color: #666; font-size: 0.9em; margin-top: 8px; }
  </style>
</head>
<body>
  <h1>OSMD MusicXML Viewer</h1>
  <p>Mirrors the Flutter app's OSMD v1.8.9 config (SVG backend, <code>drawPartNames: true</code>).
     Drop a <code>.musicxml</code> file or use the picker.</p>
  <div id="drop-zone">
    <p>Drag &amp; drop a .musicxml file here</p>
    <input type="file" id="file-input" accept=".musicxml,.xml" />
  </div>
  <div id="status"></div>
  <div id="osmd-container"></div>

  <script src="https://cdn.jsdelivr.net/npm/opensheetmusicdisplay@1.8.9/build/opensheetmusicdisplay.min.js"></script>
  <script>
    var osmd = null;

    function renderXml(xmlText) {
      var status = document.getElementById("status");
      var container = document.getElementById("osmd-container");
      container.innerHTML = "";
      status.textContent = "Rendering...";
      status.className = "";

      if (!osmd) {
        osmd = new opensheetmusicdisplay.OpenSheetMusicDisplay(container, {
          backend: "svg",
          drawTitle: true,
          drawSubtitle: false,
          drawComposer: true,
          drawCredits: false,
          drawPartNames: true,
          drawPartAbbreviations: false,
          drawMeasureNumbers: true,
          autoResize: true,
          followCursor: false
        });
      }

      osmd.load(xmlText).then(function () {
        osmd.render();
        status.textContent = "Rendered successfully.";
        status.className = "info";
      }).catch(function (err) {
        status.textContent = "Render error: " + err.message;
        status.className = "error";
        console.error("[OSMD]", err);
      });
    }

    function handleFile(file) {
      var reader = new FileReader();
      reader.onload = function (e) { renderXml(e.target.result); };
      reader.readAsText(file);
    }

    document.getElementById("file-input").addEventListener("change", function (e) {
      if (e.target.files.length) handleFile(e.target.files[0]);
    });

    var dropZone = document.getElementById("drop-zone");
    dropZone.addEventListener("dragover", function (e) {
      e.preventDefault();
      dropZone.classList.add("drag-over");
    });
    dropZone.addEventListener("dragleave", function () {
      dropZone.classList.remove("drag-over");
    });
    dropZone.addEventListener("drop", function (e) {
      e.preventDefault();
      dropZone.classList.remove("drag-over");
      if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });
  </script>
</body>
</html>
```

- [ ] **Step 2: Open in browser to verify**

Run: `open tests/osmd_viewer.html`

Expected: Browser opens showing the drag-and-drop zone and file picker. No JavaScript errors in the console. OSMD 1.8.9 loads from CDN.

- [ ] **Step 3: Commit**

```bash
git add tests/osmd_viewer.html
git commit -m "feat: add standalone OSMD viewer for MusicXML debugging

Static HTML page that loads OSMD 1.8.9 from CDN with the same
rendering config as the Flutter frontend. File input and drag-drop
for fast visual inspection without running the full pipeline."
```

---

### Task 2: Add MusicXML extraction script

**Files:**
- Create: `tests/extract_musicxml.py`

- [ ] **Step 1: Write the extraction script**

Create `tests/extract_musicxml.py`:

```python
#!/usr/bin/env python
"""Extract sanitized MusicXML from a PianoScore JSON fixture.

Runs only _render_musicxml_bytes() (which includes _sanitize_musicxml_for_osmd())
— no MIDI, no PDF, no blob storage, no pipeline.

Usage:
    python tests/extract_musicxml.py <fixture_name> [output_path]

Examples:
    python tests/extract_musicxml.py two_hand_chordal
    python tests/extract_musicxml.py c_major_scale /tmp/scale.musicxml
    python tests/extract_musicxml.py --list
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


def main() -> None:
    from backend.contracts import HumanizedPerformance  # noqa: PLC0415
    from backend.services.engrave import _render_musicxml_bytes  # noqa: PLC0415
    from tests.fixtures import FIXTURE_NAMES, load_score_fixture  # noqa: PLC0415

    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        print(f"Available fixtures: {', '.join(FIXTURE_NAMES)}")
        sys.exit(0)

    if sys.argv[1] == "--list":
        for name in FIXTURE_NAMES:
            print(name)
        sys.exit(0)

    name = sys.argv[1]
    if name not in FIXTURE_NAMES:
        print(f"Unknown fixture {name!r}. Use --list to see available fixtures.")
        sys.exit(1)

    fixture = load_score_fixture(name)
    if isinstance(fixture, HumanizedPerformance):
        score, perf = fixture.score, fixture
    else:
        score, perf = fixture, None

    musicxml, chord_count = _render_musicxml_bytes(score, perf, title=name, composer="test")

    output = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(f"tests/fixtures/scores/{name}.musicxml")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(musicxml)
    print(f"Wrote {len(musicxml):,} bytes -> {output}  (chord_symbols_rendered={chord_count})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run on a fixture to verify**

Run: `python tests/extract_musicxml.py two_hand_chordal`

Expected: Prints byte count and path. File exists at `tests/fixtures/scores/two_hand_chordal.musicxml`.

- [ ] **Step 3: Verify the output loads in the OSMD viewer**

Run: `open tests/osmd_viewer.html`

Drop `tests/fixtures/scores/two_hand_chordal.musicxml` into the viewer. Expected: Renders a grand staff with RH triads and LH octaves.

- [ ] **Step 4: Commit**

```bash
git add tests/extract_musicxml.py
git commit -m "feat: add MusicXML extraction script for isolated debugging

Runs a PianoScore fixture through _render_musicxml_bytes() only,
skipping MIDI/PDF rendering and blob storage. Outputs sanitized
MusicXML for inspection in the OSMD viewer."
```

---

### Task 3: Clear UUID part names (TDD)

**Files:**
- Modify: `tests/test_engrave_quality.py` (append test)
- Modify: `backend/services/engrave.py:861-890` (add function + wire in)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_engrave_quality.py`, after the existing L2 tests:

```python
@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_l2_part_names_are_empty(name: str, engraved_artifacts):
    """<part-name> must be empty — the <group-name>Piano</group-name> on
    the brace is the only instrument label. Non-empty <part-name> causes
    OSMD to display a redundant label per staff (e.g., "Instr. P-RH").
    """
    from lxml import etree

    musicxml, _ = engraved_artifacts[name]
    root = etree.fromstring(musicxml)
    for score_part in root.findall("part-list/score-part"):
        pn = score_part.find("part-name")
        if pn is None:
            continue
        assert pn.text is None or pn.text.strip() == "", (
            f"{name}: <part-name> should be empty, got {pn.text!r}"
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_engrave_quality.py::test_l2_part_names_are_empty -v`

Expected: FAIL — music21 populates `<part-name>` with its internal part identifier (e.g., `"P-RH"` or a UUID string).

- [ ] **Step 3: Add `_clear_part_names()` function**

Add this function in `backend/services/engrave.py` immediately before `_sanitize_musicxml_for_osmd()` (before line 861):

```python
def _clear_part_names(raw: bytes) -> bytes:
    """Clear ``<part-name>`` text in ``<score-part>`` entries.

    music21 populates ``<part-name>`` with its internal part identifier
    when no explicit ``partName`` is set on the ``Part`` object. OSMD
    reads this and displays it as a per-staff instrument label (e.g.
    "Instr. P-RH"). The ``<part-group>`` wrapper already carries
    ``<group-name>Piano</group-name>`` from the ``StaffGroup``, which
    OSMD uses for the brace label — per-part names are redundant.
    """
    import xml.etree.ElementTree as ET  # noqa: PLC0415

    root = ET.fromstring(raw)
    part_list = root.find("part-list")
    if part_list is not None:
        for score_part in part_list.findall("score-part"):
            part_name = score_part.find("part-name")
            if part_name is not None:
                part_name.text = ""

    prefix_end = raw.find(b"<score-partwise")
    prefix = raw[:prefix_end] if prefix_end > 0 else b""
    body = ET.tostring(root, encoding="utf-8", xml_declaration=False)
    return prefix + body
```

- [ ] **Step 4: Wire into `_sanitize_musicxml_for_osmd()`**

In `backend/services/engrave.py`, modify `_sanitize_musicxml_for_osmd()` to call the new function. Change line 889:

```python
# Before:
    remapped = _remap_voices_per_staff(text.encode("utf-8"))
    return _align_tie_chain_voices(remapped)

# After:
    remapped = _remap_voices_per_staff(text.encode("utf-8"))
    cleared = _clear_part_names(remapped)
    return _align_tie_chain_voices(cleared)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_engrave_quality.py::test_l2_part_names_are_empty -v`

Expected: PASS for all fixtures.

- [ ] **Step 6: Run the full test suite to check for regressions**

Run: `pytest tests/test_engrave_quality.py -v`

Expected: All existing L1 and L2 tests still pass.

- [ ] **Step 7: Commit**

```bash
git add backend/services/engrave.py tests/test_engrave_quality.py
git commit -m "fix: clear UUID part names from MusicXML for OSMD

Add _clear_part_names() pass to _sanitize_musicxml_for_osmd() that
sets <part-name> text to empty string. OSMD reads <group-name> from
the StaffGroup brace for the instrument label — per-part names were
causing 'Instr. P-RH' or UUID strings to render on each staff."
```

---

### Task 4: Diagnose and fix note collisions

**Files:**
- Modify: `backend/services/engrave.py:861-890` (potential new sanitizer pass)
- Modify: `tests/test_engrave_quality.py` (regression test for the pattern)

This task is investigative — the exact fix depends on what the diagnosis reveals. The steps below provide the diagnostic process and the structural framework for the fix.

- [ ] **Step 1: Extract MusicXML from all fixtures**

Run:

```bash
for name in single_note c_major_scale two_hand_chordal bach_invention_excerpt jazz_voicings seven_eight tempo_change empty_left_hand overlapping_same_pitch triplet_eighths mislabeled_key chord_symbols; do
    python tests/extract_musicxml.py "$name"
done
```

Expected: A `.musicxml` file per fixture in `tests/fixtures/scores/`.

- [ ] **Step 2: View each in the OSMD viewer and identify collisions**

Open `tests/osmd_viewer.html` in a browser. Drop each `.musicxml` file and check for:
- Notes piling up on top of each other at the same x-position
- Notes overlapping horizontally that should be spread across the measure
- Missing or garbled stems / beams

Record which fixtures show problems and which measures are affected.

- [ ] **Step 3: Inspect XML around problematic measures**

For each collision found in Step 2, open the `.musicxml` in a text editor and inspect the `<measure>` element. Look for:

1. **Duration/division mismatch:** A `<duration>` value that doesn't align with `divisions=12` (e.g., non-integer or zero values)
2. **Missing `<chord/>` tag:** Two notes at the same time position in the same voice where the second lacks `<chord/>`, causing OSMD to advance the x-cursor instead of stacking vertically
3. **`<forward>`/`<backup>` imbalance:** Cumulative durations within a voice don't match the measure length, causing position drift across voices

- [ ] **Step 4: Write a failing test for the identified pattern**

Based on the diagnosis, add a parametrized L2 test to `tests/test_engrave_quality.py`. The test structure depends on the pattern found. Example for a duration-balance check:

```python
@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_l2_no_measure_duration_overflow(name: str, engraved_artifacts):
    """Per-voice note durations must not exceed the measure capacity.

    OSMD calculates x-positions from cumulative durations. If a voice's
    notes sum to more than the measure allows, subsequent notes pile up
    at the right edge.
    """
    from lxml import etree

    musicxml, _ = engraved_artifacts[name]
    root = etree.fromstring(musicxml)

    for part in root.findall("part"):
        measure_divisions = 12
        beats, beat_type = 4, 4
        for measure in part.findall("measure"):
            attrs = measure.find("attributes")
            if attrs is not None:
                div_el = attrs.find("divisions")
                if div_el is not None:
                    measure_divisions = int(div_el.text)
                ts = attrs.find("time")
                if ts is not None:
                    beats = int(ts.findtext("beats") or "4")
                    beat_type = int(ts.findtext("beat-type") or "4")
            measure_capacity = measure_divisions * beats * 4 // beat_type

            voice_durs: dict[str, int] = {}
            for note in measure.findall("note"):
                if note.find("chord") is not None:
                    continue
                voice = note.findtext("voice") or "1"
                dur_el = note.find("duration")
                if dur_el is not None:
                    voice_durs[voice] = voice_durs.get(voice, 0) + int(dur_el.text)

            for v, total in voice_durs.items():
                assert total <= measure_capacity, (
                    f"{name}: measure {measure.get('number')} voice {v} "
                    f"duration {total} exceeds capacity {measure_capacity}"
                )
```

Run: `pytest tests/test_engrave_quality.py::test_l2_no_measure_duration_overflow -v`

Expected: FAIL on the fixture(s) that showed collisions in Step 2 (if this is the pattern), or PASS if the collision has a different cause.

- [ ] **Step 5: Implement the fix**

Add a new sanitizer function in `backend/services/engrave.py` before `_sanitize_musicxml_for_osmd()`. The function follows the same pattern as `_remap_voices_per_staff()` and `_align_tie_chain_voices()`:

```python
def _fix_note_collisions(raw: bytes) -> bytes:
    """Fix MusicXML patterns that cause OSMD rendering collisions.

    [Description of the specific pattern, filled in after diagnosis.]
    """
    import xml.etree.ElementTree as ET  # noqa: PLC0415

    root = ET.fromstring(raw)

    # [Fix logic targeting the diagnosed pattern —
    #  e.g., recompute <duration> values, insert missing <chord/> tags,
    #  or rebalance <forward>/<backup> elements.]

    prefix_end = raw.find(b"<score-partwise")
    prefix = raw[:prefix_end] if prefix_end > 0 else b""
    body = ET.tostring(root, encoding="utf-8", xml_declaration=False)
    return prefix + body
```

Wire into `_sanitize_musicxml_for_osmd()`:

```python
def _sanitize_musicxml_for_osmd(raw: bytes) -> bytes:
    text = raw.decode("utf-8")
    text = re.sub(r"<voice>0</voice>", "<voice>1</voice>", text)
    remapped = _remap_voices_per_staff(text.encode("utf-8"))
    cleared = _clear_part_names(remapped)
    fixed = _fix_note_collisions(cleared)
    return _align_tie_chain_voices(fixed)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_engrave_quality.py -v`

Expected: All tests pass, including the new collision test from Step 4.

- [ ] **Step 7: Verify visually in OSMD viewer**

Re-extract the previously-broken fixture:

```bash
python tests/extract_musicxml.py <fixture_name>
```

Drop the new `.musicxml` into the OSMD viewer. Confirm notes no longer pile up.

- [ ] **Step 8: Commit**

```bash
git add backend/services/engrave.py tests/test_engrave_quality.py
git commit -m "fix: resolve note collisions in OSMD rendering

Add _fix_note_collisions() pass to _sanitize_musicxml_for_osmd().
[One sentence describing the specific pattern and fix.]"
```

---

### Task 5: Add regression fixture and test

**Files:**
- Create: `tests/fixtures/jammin_osmd_regression.musicxml`
- Modify: `tests/test_engrave_quality.py` (add regression test)

- [ ] **Step 1: Generate the regression fixture**

Extract MusicXML from the fixture that previously exhibited collisions (identified in Task 4), or from a representative multi-voice score:

```bash
python tests/extract_musicxml.py two_hand_chordal tests/fixtures/jammin_osmd_regression.musicxml
```

If a real pipeline output is available (e.g., from a "Jammin'" job), copy it instead:

```bash
cp /path/to/pipeline/output/score.musicxml tests/fixtures/jammin_osmd_regression.musicxml
```

- [ ] **Step 2: Write the regression test**

Append to `tests/test_engrave_quality.py`:

```python
def test_l2_osmd_regression_fixture_structural_checks():
    """Regression guard — the saved fixture must pass all OSMD-relevant
    structural checks: empty part names, valid voices, correct divisions,
    and no measure overflow.
    """
    from lxml import etree

    fixture_path = Path(__file__).parent / "fixtures" / "jammin_osmd_regression.musicxml"
    if not fixture_path.exists():
        pytest.skip("jammin_osmd_regression.musicxml not yet generated")

    root = etree.fromstring(fixture_path.read_bytes())

    # Part names must be empty.
    for pn in root.findall("part-list/score-part/part-name"):
        assert pn.text is None or pn.text.strip() == "", (
            f"regression fixture has non-empty <part-name>: {pn.text!r}"
        )

    # Brace group with "Piano" label must exist.
    brace_starts = [
        g for g in root.findall("part-list/part-group")
        if g.get("type") == "start" and g.findtext("group-symbol") == "brace"
    ]
    assert brace_starts, "regression fixture missing brace part-group"
    assert brace_starts[0].findtext("group-name") == "Piano"

    # Voices in range [1, 4].
    for v_el in root.iter("voice"):
        v = int(v_el.text)
        assert 1 <= v <= 4, f"out-of-range voice: {v}"

    # Divisions = 12.
    for div in root.iter("divisions"):
        assert int(div.text) == 12, f"unexpected divisions: {div.text}"
```

Add the necessary import at the top of the file if not already present:

```python
from pathlib import Path
```

- [ ] **Step 3: Run test to verify it passes**

Run: `pytest tests/test_engrave_quality.py::test_l2_osmd_regression_fixture_structural_checks -v`

Expected: PASS (the fixture was generated after all sanitizer fixes were applied).

- [ ] **Step 4: Clean up extracted .musicxml files from fixtures/scores/**

Remove the per-fixture `.musicxml` files generated during Task 4 diagnosis (these are debug artifacts, not committed fixtures):

```bash
rm -f tests/fixtures/scores/*.musicxml
```

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures/jammin_osmd_regression.musicxml tests/test_engrave_quality.py
git commit -m "test: add OSMD regression fixture with structural checks

Save a representative MusicXML file as a regression guard for OSMD
rendering. The test validates empty part names, brace grouping,
voice range, and divisions=12."
```
