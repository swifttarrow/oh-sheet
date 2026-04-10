/// Web-specific interactive sheet music viewer: OSMD + Tone.js + cursor sync.
///
/// Architecture (matching the proven TuneChat pattern):
///   1. OSMD renders MusicXML as interactive SVG
///   2. @tonejs/midi parses MIDI → flat note list → 20ms chord groups
///   3. Tone.js PolySynth plays groups on Transport
///   4. Tone.getDraw().schedule() fires cursor.next() on the visual frame
///      (not the audio thread) so highlight matches what you hear
///   5. Falls back to PDF iframe if OSMD fails
library;

import 'dart:js_interop';
import 'dart:ui_web' as ui_web;

import 'package:flutter/material.dart';
import 'package:web/web.dart' as web;

class WebSheetMusicViewer extends StatefulWidget {
  const WebSheetMusicViewer({
    super.key,
    required this.musicxmlUrl,
    required this.midiUrl,
  });
  final String musicxmlUrl;
  final String midiUrl;

  @override
  State<WebSheetMusicViewer> createState() => _WebSheetMusicViewerState();
}

class _WebSheetMusicViewerState extends State<WebSheetMusicViewer> {
  late final String _viewType;

  @override
  void initState() {
    super.initState();
    final hash = '${widget.musicxmlUrl}-${widget.midiUrl}'.hashCode.abs();
    _viewType = 'osmd-viewer-$hash';

    ui_web.platformViewRegistry.registerViewFactory(_viewType, (int viewId) {
      final container = web.document.createElement('div') as web.HTMLElement;
      container.style.width = '100%';
      container.style.height = '100%';
      container.style.display = 'flex';
      container.style.flexDirection = 'column';
      container.style.backgroundColor = 'white';
      container.style.overflow = 'hidden';

      final sheetDiv = web.document.createElement('div') as web.HTMLElement;
      sheetDiv.id = 'osmd-sheet-$hash';
      sheetDiv.style.flex = '1';
      sheetDiv.style.overflow = 'auto';
      sheetDiv.style.padding = '8px';
      sheetDiv.style.minHeight = '0';

      final loadingText = web.document.createElement('p') as web.HTMLElement;
      loadingText.textContent = 'Loading sheet music...';
      loadingText.style.textAlign = 'center';
      loadingText.style.color = '#999';
      loadingText.style.padding = '40px';
      sheetDiv.appendChild(loadingText);

      final controlsDiv = web.document.createElement('div') as web.HTMLElement;
      controlsDiv.id = 'osmd-controls-$hash';
      controlsDiv.style.borderTop = '1px solid #e0e0e0';
      controlsDiv.style.padding = '8px 16px';
      controlsDiv.style.display = 'flex';
      controlsDiv.style.alignItems = 'center';
      controlsDiv.style.gap = '12px';
      controlsDiv.style.flexShrink = '0';
      controlsDiv.style.backgroundColor = '#fafafa';

      final playBtn = web.document.createElement('button') as web.HTMLElement;
      playBtn.id = 'osmd-play-$hash';
      playBtn.textContent = '\u25B6 Play';
      playBtn.style.padding = '6px 16px';
      playBtn.style.border = '2px solid #2D3436';
      playBtn.style.borderRadius = '20px';
      playBtn.style.backgroundColor = '#4DB6AC';
      playBtn.style.color = 'white';
      playBtn.style.fontWeight = '700';
      playBtn.style.cursor = 'pointer';
      playBtn.style.fontSize = '14px';

      final stopBtn = web.document.createElement('button') as web.HTMLElement;
      stopBtn.id = 'osmd-stop-$hash';
      stopBtn.textContent = '\u23F9 Stop';
      stopBtn.style.padding = '6px 16px';
      stopBtn.style.border = '2px solid #2D3436';
      stopBtn.style.borderRadius = '20px';
      stopBtn.style.backgroundColor = 'white';
      stopBtn.style.cursor = 'pointer';
      stopBtn.style.fontSize = '14px';

      final timeSpan = web.document.createElement('span') as web.HTMLElement;
      timeSpan.id = 'osmd-time-$hash';
      timeSpan.textContent = '0:00';
      timeSpan.style.color = '#636E72';
      timeSpan.style.fontSize = '13px';
      timeSpan.style.fontFamily = 'monospace';

      controlsDiv.appendChild(playBtn);
      controlsDiv.appendChild(stopBtn);
      controlsDiv.appendChild(timeSpan);

      container.appendChild(sheetDiv);
      container.appendChild(controlsDiv);

      _initViewer(hash);

      return container;
    });
  }

  void _initViewer(int hash) {
    final musicxmlUrl = widget.musicxmlUrl;
    final midiUrl = widget.midiUrl;
    final sheetId = 'osmd-sheet-$hash';
    final controlsId = 'osmd-controls-$hash';
    final playBtnId = 'osmd-play-$hash';
    final stopBtnId = 'osmd-stop-$hash';
    final timeId = 'osmd-time-$hash';

    // The init script uses the proven TuneChat architecture:
    // 1. Flatten MIDI notes → sort by time → group by 20ms threshold
    // 2. Schedule each group on Transport with triggerAttackRelease
    // 3. Use Tone.getDraw().schedule() for cursor sync (visual frame time)
    final initScript = '''
(function() {
  var attempts = 0;
  function tryInit() {
    attempts++;
    var container = document.getElementById("$sheetId");
    var controlsEl = document.getElementById("$controlsId");
    var playBtn = document.getElementById("$playBtnId");
    var stopBtn = document.getElementById("$stopBtnId");
    var timeEl = document.getElementById("$timeId");

    if (!container || !window.Tone || !window.Midi || attempts > 30) {
      if (attempts <= 30) { setTimeout(tryInit, 300); return; }
      console.error("[OSMD] Gave up waiting for deps");
      if (controlsEl) controlsEl.style.display = "none";
      if (container) {
        while (container.firstChild) container.removeChild(container.firstChild);
        var fallback = document.createElement("iframe");
        fallback.src = "$musicxmlUrl".replace("/musicxml", "/pdf") + "?inline=true";
        fallback.style.cssText = "width:100%;height:100%;border:none";
        container.appendChild(fallback);
      }
      return;
    }

    console.log("[OSMD] Init attempt", attempts,
      "OSMD lib:", !!window.opensheetmusicdisplay);

    // ─── Step 1: Load MIDI + MusicXML in parallel ───
    var midiPromise = fetch("$midiUrl")
      .then(function(r) { return r.arrayBuffer(); });
    var xmlPromise = fetch("$musicxmlUrl")
      .then(function(r) { return r.text(); });

    Promise.all([midiPromise, xmlPromise]).then(function(results) {
      var midiBuffer = results[0];
      var xmlText = results[1];

      // ─── Step 2: Parse MIDI → flatten → group by 20ms ───
      var midi = new Midi(midiBuffer);
      console.log("[OSMD] MIDI:", midi.tracks.length, "tracks,",
        midi.duration.toFixed(1), "sec");

      // Thresholds for filtering out silent/ghost notes before playback.
      // Keep in sync with backend MIN_NOTE_DUR in engrave.py.
      // - velocity is normalized 0.0–1.0 from @tonejs/midi
      // - duration is in seconds
      var MIN_VELOCITY = 0.05;        // ~6/127 MIDI — inaudible below this
      var MIN_DURATION_SEC = 0.03;    // matches engrave.py MIN_NOTE_DUR

      var allNotes = [];
      var skippedSilent = 0;
      for (var t = 0; t < midi.tracks.length; t++) {
        var track = midi.tracks[t];
        for (var n = 0; n < track.notes.length; n++) {
          var note = track.notes[n];
          // Skip inaudible / zero-length notes so we don't play "rests"
          if (note.velocity < MIN_VELOCITY || note.duration < MIN_DURATION_SEC) {
            skippedSilent++;
            continue;
          }
          allNotes.push({
            name: note.name,
            time: note.time,
            duration: note.duration,
            velocity: note.velocity
          });
        }
      }
      allNotes.sort(function(a, b) { return a.time - b.time; });
      if (skippedSilent > 0) {
        console.log("[OSMD] Skipped", skippedSilent, "silent/short notes");
      }

      if (allNotes.length === 0) {
        console.warn("[OSMD] No audible notes in MIDI");
        return;
      }

      // Group notes within 20ms as simultaneous (chords)
      var groups = [];
      var currentGroup = { time: allNotes[0].time, notes: [allNotes[0]] };
      for (var i = 1; i < allNotes.length; i++) {
        if (allNotes[i].time - currentGroup.time < 0.02) {
          currentGroup.notes.push(allNotes[i]);
        } else {
          groups.push(currentGroup);
          currentGroup = { time: allNotes[i].time, notes: [allNotes[i]] };
        }
      }
      groups.push(currentGroup);
      console.log("[OSMD] Note groups:", groups.length,
        "(from", allNotes.length, "notes)");

      // ─── Step 3: Try OSMD rendering ───
      var osmdOk = false;
      var cursor = null;

      if (window.opensheetmusicdisplay) {
        try {
          var osmd = new opensheetmusicdisplay.OpenSheetMusicDisplay(container, {
            backend: "svg",
            drawTitle: false,
            drawSubtitle: false,
            drawComposer: false,
            drawCredits: false,
            drawPartNames: true,
            drawPartAbbreviations: false,
            drawMeasureNumbers: true,
            autoResize: false,
            followCursor: true,
            cursorsOptions: [{type: 0, color: "#FF3B30", alpha: 0.6, follow: true}]
          });

          // loadSync-style: load returns a promise
          return osmd.load(xmlText).then(function() {
            osmd.render();
            cursor = osmd.cursor;
            cursor.show();
            cursor.reset();
            osmdOk = true;
            console.log("[OSMD] Render succeeded! Cursor active.");
            wirePlayback();
          }).catch(function(osmdErr) {
            console.warn("[OSMD] Render failed:", osmdErr.message);
            fallbackToPdf();
            wirePlayback();
          });
        } catch(e) {
          console.warn("[OSMD] Setup failed:", e.message);
          fallbackToPdf();
          wirePlayback();
          return;
        }
      } else {
        console.log("[OSMD] OSMD lib not loaded, using PDF");
        fallbackToPdf();
        wirePlayback();
        return;
      }

      function fallbackToPdf() {
        while (container.firstChild) container.removeChild(container.firstChild);
        var iframe = document.createElement("iframe");
        iframe.src = "$musicxmlUrl".replace("/musicxml", "/pdf") + "?inline=true";
        iframe.style.cssText = "width:100%;height:100%;border:none";
        container.style.overflow = "hidden";
        container.appendChild(iframe);
      }

      // ─── Step 4: Wire Tone.js playback (works with or without OSMD) ───
      function wirePlayback() {
        var synth = new Tone.PolySynth(Tone.Synth, {
          maxPolyphony: 16,
          voice: Tone.Synth,
          options: {
            envelope: { attack: 0.02, decay: 0.1, sustain: 0.3, release: 0.8 }
          }
        }).toDestination();
        synth.volume.value = -6;

        var isPlaying = false;
        var scheduledIds = [];
        var groupIndex = 0;

        function formatTime(sec) {
          var m = Math.floor(sec / 60);
          var s = Math.floor(sec % 60);
          return m + ":" + (s < 10 ? "0" : "") + s;
        }

        function updateTimeDisplay() {
          if (!isPlaying) return;
          var t = Tone.getTransport().seconds;
          if (timeEl) timeEl.textContent = formatTime(t) + " / " + formatTime(midi.duration);
          requestAnimationFrame(updateTimeDisplay);
        }

        function scheduleGroups() {
          scheduledIds.forEach(function(id) { Tone.getTransport().clear(id); });
          scheduledIds = [];
          groupIndex = 0;

          for (var g = 0; g < groups.length; g++) {
            (function(group, idx) {
              var id = Tone.getTransport().schedule(function(time) {
                // Play all notes in the group
                for (var n = 0; n < group.notes.length; n++) {
                  var note = group.notes[n];
                  synth.triggerAttackRelease(
                    note.name, note.duration, time, note.velocity
                  );
                }
                // Advance cursor on the VISUAL frame (not audio thread)
                if (osmdOk && cursor) {
                  Tone.getDraw().schedule(function() {
                    try { cursor.next(); } catch(e) { osmdOk = false; }
                    groupIndex = idx + 1;
                  }, time);
                }
              }, group.time);
              scheduledIds.push(id);
            })(groups[g], g);
          }

          // End event
          var lastGroup = groups[groups.length - 1];
          var lastDur = Math.max.apply(null, lastGroup.notes.map(function(n) { return n.duration; }));
          var endTime = lastGroup.time + lastDur + 0.5;

          Tone.getTransport().schedule(function(time) {
            Tone.getDraw().schedule(function() {
              stopPlayback();
            }, time);
          }, endTime);
        }

        function startPlayback() {
          if (isPlaying) return;
          Tone.start().then(function() {
            isPlaying = true;
            groupIndex = 0;
            if (osmdOk && cursor) {
              try { cursor.reset(); cursor.show(); } catch(e) {}
            }
            var bpm = (midi.header.tempos && midi.header.tempos.length > 0)
              ? midi.header.tempos[0].bpm : 120;
            Tone.getTransport().bpm.value = bpm;
            Tone.getTransport().seconds = 0;
            scheduleGroups();
            Tone.getTransport().start("+0.1");
            if (playBtn) playBtn.textContent = "\u23F8 Pause";
            updateTimeDisplay();
          });
        }

        function pausePlayback() {
          isPlaying = false;
          Tone.getTransport().pause();
          if (playBtn) playBtn.textContent = "\u25B6 Play";
        }

        function stopPlayback() {
          isPlaying = false;
          Tone.getTransport().stop();
          Tone.getTransport().cancel();
          scheduledIds = [];
          groupIndex = 0;
          if (osmdOk && cursor) {
            try { cursor.reset(); cursor.show(); } catch(e) {}
          }
          if (playBtn) playBtn.textContent = "\u25B6 Play";
          if (timeEl) timeEl.textContent = "0:00 / " + formatTime(midi.duration);
        }

        // Wire button events
        if (playBtn) {
          playBtn.addEventListener("click", function() {
            if (isPlaying) pausePlayback();
            else startPlayback();
          });
        }
        if (stopBtn) {
          stopBtn.addEventListener("click", stopPlayback);
        }
        if (timeEl) {
          timeEl.textContent = "0:00 / " + formatTime(midi.duration);
        }

        console.log("[OSMD] Playback wired. OSMD cursor:", osmdOk);
      }
    }).catch(function(err) {
      console.error("[OSMD] Fatal:", err);
      if (controlsEl) controlsEl.style.display = "none";
      while (container.firstChild) container.removeChild(container.firstChild);
      var errP = document.createElement("p");
      errP.textContent = "Failed to load sheet music";
      errP.style.cssText = "color:#999;text-align:center;padding:40px";
      container.appendChild(errP);
    });
  }
  setTimeout(tryInit, 300);
})();
''';

    _jsEval(initScript);
  }

  void _jsEval(String code) {
    _evalJS(code.toJS);
  }

  @override
  Widget build(BuildContext context) {
    return HtmlElementView(viewType: _viewType);
  }
}

@JS('eval')
external void _evalJS(JSString code);
