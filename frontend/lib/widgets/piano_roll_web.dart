/// Web-specific piano roll: canvas-based note visualization with Y-axis labels.
///
/// Reads playback position from Tone.getTransport().seconds so it syncs
/// with the Sheet Music section's Tone.js player. No own audio engine.
///
/// Drawing layers:
///   1. Background stripes (white/grey for white/black keys)
///   2. Note rectangles (teal for RH, orange for LH, split at middle C)
///   3. Y-axis labels (C3, D3... pinned on left edge)
///   4. Red playback line (moves with transport, auto-scrolls view)
library;

import 'dart:js_interop';
import 'dart:ui_web' as ui_web;

import 'package:flutter/material.dart';
import 'package:web/web.dart' as web;

class WebPianoRoll extends StatefulWidget {
  const WebPianoRoll({super.key, required this.midiUrl});
  final String midiUrl;

  @override
  State<WebPianoRoll> createState() => _WebPianoRollState();
}

class _WebPianoRollState extends State<WebPianoRoll> {
  late final String _viewType;

  @override
  void initState() {
    super.initState();
    final hash = widget.midiUrl.hashCode.abs();
    _viewType = 'piano-roll-$hash';

    ui_web.platformViewRegistry.registerViewFactory(_viewType, (int viewId) {
      final container = web.document.createElement('div') as web.HTMLElement;
      container.id = 'piano-roll-container-$hash';
      container.style.width = '100%';
      container.style.height = '100%';
      container.style.position = 'relative';
      container.style.overflow = 'hidden';
      container.style.backgroundColor = '#1a1a2e';

      // Loading text
      final loading = web.document.createElement('p') as web.HTMLElement;
      loading.textContent = 'Loading piano roll...';
      loading.style.color = '#666';
      loading.style.textAlign = 'center';
      loading.style.padding = '40px';
      container.appendChild(loading);

      _initPianoRoll(hash);
      return container;
    });
  }

  void _initPianoRoll(int hash) {
    final midiUrl = widget.midiUrl;
    final containerId = 'piano-roll-container-$hash';

    final initScript = '''
(function() {
  var attempts = 0;
  function tryInit() {
    attempts++;
    var container = document.getElementById("$containerId");
    if (!container || !window.Midi || !window.Tone || attempts > 30) {
      if (attempts <= 30) { setTimeout(tryInit, 300); return; }
      if (container) container.textContent = "Piano roll failed to load";
      return;
    }

    fetch("$midiUrl")
      .then(function(r) { return r.arrayBuffer(); })
      .then(function(buf) {
        var midi = new Midi(buf);

        // Flatten all notes
        var allNotes = [];
        for (var t = 0; t < midi.tracks.length; t++) {
          for (var n = 0; n < midi.tracks[t].notes.length; n++) {
            var note = midi.tracks[t].notes[n];
            allNotes.push({
              midi: note.midi,
              name: note.name,
              time: note.time,
              duration: note.duration,
              velocity: note.velocity
            });
          }
        }
        if (allNotes.length === 0) {
          container.textContent = "No notes to display";
          return;
        }

        // Calculate pitch range
        var minPitch = 127, maxPitch = 0;
        var maxTime = 0;
        for (var i = 0; i < allNotes.length; i++) {
          if (allNotes[i].midi < minPitch) minPitch = allNotes[i].midi;
          if (allNotes[i].midi > maxPitch) maxPitch = allNotes[i].midi;
          var end = allNotes[i].time + allNotes[i].duration;
          if (end > maxTime) maxTime = end;
        }
        // Add padding
        minPitch = Math.max(0, minPitch - 2);
        maxPitch = Math.min(127, maxPitch + 2);
        var pitchRange = maxPitch - minPitch + 1;

        // Layout constants
        var labelWidth = 44;
        var rowHeight = 14;
        var pixelsPerSecond = 80;
        var canvasHeight = pitchRange * rowHeight;
        var canvasWidth = Math.max(container.clientWidth, maxTime * pixelsPerSecond + labelWidth + 100);

        // Clear container
        while (container.firstChild) container.removeChild(container.firstChild);

        // Scrollable wrapper
        var scrollDiv = document.createElement("div");
        scrollDiv.style.width = "100%";
        scrollDiv.style.height = "100%";
        scrollDiv.style.overflow = "auto";
        scrollDiv.style.position = "relative";

        // Canvas
        var canvas = document.createElement("canvas");
        canvas.width = canvasWidth;
        canvas.height = Math.max(canvasHeight, container.clientHeight);
        canvas.style.display = "block";
        var ctx = canvas.getContext("2d");

        scrollDiv.appendChild(canvas);
        container.appendChild(scrollDiv);

        // Note name helper
        var noteNames = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"];
        function pitchName(p) {
          return noteNames[p % 12] + Math.floor(p / 12 - 1);
        }
        function isBlackKey(p) {
          var n = p % 12;
          return n === 1 || n === 3 || n === 6 || n === 8 || n === 10;
        }

        function draw(currentTime) {
          var w = canvas.width;
          var h = canvas.height;
          ctx.clearRect(0, 0, w, h);

          // --- Background stripes ---
          for (var p = minPitch; p <= maxPitch; p++) {
            var y = (maxPitch - p) * rowHeight;
            ctx.fillStyle = isBlackKey(p) ? "#16213e" : "#1a1a2e";
            ctx.fillRect(labelWidth, y, w - labelWidth, rowHeight);

            // Subtle grid line
            ctx.strokeStyle = "rgba(255,255,255,0.05)";
            ctx.beginPath();
            ctx.moveTo(labelWidth, y);
            ctx.lineTo(w, y);
            ctx.stroke();
          }

          // --- Note rectangles ---
          for (var i = 0; i < allNotes.length; i++) {
            var note = allNotes[i];
            var x = labelWidth + note.time * pixelsPerSecond;
            var y = (maxPitch - note.midi) * rowHeight;
            var nw = Math.max(2, note.duration * pixelsPerSecond);

            // Color: teal for RH (>=60), orange for LH (<60)
            if (note.midi >= 60) {
              ctx.fillStyle = "rgba(77, 182, 172, " + (0.5 + note.velocity * 0.5) + ")";
            } else {
              ctx.fillStyle = "rgba(255, 179, 0, " + (0.5 + note.velocity * 0.5) + ")";
            }
            ctx.fillRect(x, y + 1, nw, rowHeight - 2);

            // Note border
            ctx.strokeStyle = "rgba(255,255,255,0.15)";
            ctx.strokeRect(x, y + 1, nw, rowHeight - 2);
          }

          // --- Y-axis labels (pinned) ---
          ctx.fillStyle = "#0f0f23";
          ctx.fillRect(0, 0, labelWidth, h);
          ctx.strokeStyle = "rgba(255,255,255,0.1)";
          ctx.beginPath();
          ctx.moveTo(labelWidth, 0);
          ctx.lineTo(labelWidth, h);
          ctx.stroke();

          for (var p = minPitch; p <= maxPitch; p++) {
            var y = (maxPitch - p) * rowHeight;
            // Only label natural notes (no sharps/flats) to reduce clutter
            if (!isBlackKey(p)) {
              ctx.fillStyle = "#8899aa";
              ctx.font = "10px monospace";
              ctx.textAlign = "right";
              ctx.textBaseline = "middle";
              ctx.fillText(pitchName(p), labelWidth - 4, y + rowHeight / 2);
            }
            // C notes get a brighter label
            if (p % 12 === 0) {
              ctx.fillStyle = "#4DB6AC";
              ctx.font = "bold 10px monospace";
              ctx.textAlign = "right";
              ctx.textBaseline = "middle";
              ctx.fillText(pitchName(p), labelWidth - 4, y + rowHeight / 2);
            }
          }

          // --- Beat grid lines ---
          var bpm = (midi.header.tempos && midi.header.tempos.length > 0)
            ? midi.header.tempos[0].bpm : 120;
          var beatDuration = 60 / bpm;
          for (var t = 0; t < maxTime; t += beatDuration) {
            var x = labelWidth + t * pixelsPerSecond;
            var isMeasure = Math.abs(t % (beatDuration * 4)) < 0.01;
            ctx.strokeStyle = isMeasure ? "rgba(255,255,255,0.15)" : "rgba(255,255,255,0.05)";
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, h);
            ctx.stroke();
          }

          // --- Playback line ---
          if (currentTime > 0) {
            var px = labelWidth + currentTime * pixelsPerSecond;
            ctx.strokeStyle = "#FF3B30";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(px, 0);
            ctx.lineTo(px, h);
            ctx.stroke();
            ctx.lineWidth = 1;

            // Glow effect
            ctx.shadowColor = "#FF3B30";
            ctx.shadowBlur = 8;
            ctx.strokeStyle = "rgba(255, 59, 48, 0.5)";
            ctx.beginPath();
            ctx.moveTo(px, 0);
            ctx.lineTo(px, h);
            ctx.stroke();
            ctx.shadowBlur = 0;
          }
        }

        // Initial draw
        draw(0);

        // Animation loop — reads from Tone.js transport
        var rafId = null;
        function animate() {
          var t = 0;
          try {
            if (Tone.getTransport().state === "started") {
              t = Tone.getTransport().seconds;
            }
          } catch(e) {}
          draw(t);

          // Auto-scroll to keep playback line centered
          if (t > 0) {
            var px = labelWidth + t * pixelsPerSecond;
            var viewCenter = scrollDiv.clientWidth / 2;
            var targetScroll = px - viewCenter;
            if (Math.abs(scrollDiv.scrollLeft - targetScroll) > 50) {
              scrollDiv.scrollLeft += (targetScroll - scrollDiv.scrollLeft) * 0.1;
            }
          }

          rafId = requestAnimationFrame(animate);
        }
        animate();

        console.log("[PianoRoll] Rendered", allNotes.length, "notes, pitch range",
          pitchName(minPitch), "-", pitchName(maxPitch));
      })
      .catch(function(err) {
        console.error("[PianoRoll] Failed:", err);
        container.textContent = "Could not load piano roll";
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
