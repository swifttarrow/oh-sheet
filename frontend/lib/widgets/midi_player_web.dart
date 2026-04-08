/// Web-specific MIDI player + visualizer using html-midi-player web component.
/// The <midi-visualizer> shows a piano roll where notes light up during playback.
/// It automatically syncs with the <midi-player> when they share the same src.
library;

import 'dart:ui_web' as ui_web;

import 'package:flutter/material.dart';
import 'package:web/web.dart' as web;

class WebMidiPlayer extends StatefulWidget {
  const WebMidiPlayer({super.key, required this.midiUrl});
  final String midiUrl;

  @override
  State<WebMidiPlayer> createState() => _WebMidiPlayerState();
}

class _WebMidiPlayerState extends State<WebMidiPlayer> {
  late final String _playerViewType;

  @override
  void initState() {
    super.initState();
    final hash = widget.midiUrl.hashCode;

    _playerViewType = 'midi-player-$hash';
    ui_web.platformViewRegistry.registerViewFactory(_playerViewType, (int viewId) {
      final container = web.document.createElement('div') as web.HTMLElement;
      container.style.width = '100%';
      container.style.height = '100%';
      container.style.display = 'flex';
      container.style.flexDirection = 'column';

      // Player controls (play/pause/scrub)
      final player = web.document.createElement('midi-player') as web.HTMLElement;
      player.setAttribute('src', widget.midiUrl);
      player.setAttribute('sound-font', '');
      player.id = 'midi-player-$hash';
      player.style.width = '100%';

      // Piano roll visualizer — notes light up during playback
      final visualizer = web.document.createElement('midi-visualizer') as web.HTMLElement;
      visualizer.setAttribute('src', widget.midiUrl);
      visualizer.setAttribute('type', 'piano-roll');
      visualizer.style.width = '100%';
      visualizer.style.flexGrow = '1';
      visualizer.style.minHeight = '120px';
      visualizer.style.overflow = 'auto';

      container.appendChild(player);
      container.appendChild(visualizer);
      return container;
    });
  }

  @override
  Widget build(BuildContext context) {
    return HtmlElementView(viewType: _playerViewType);
  }
}
