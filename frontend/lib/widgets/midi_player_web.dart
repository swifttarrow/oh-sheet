/// Web-specific MIDI player using html-midi-player web component.
/// This file is only imported on web via conditional import.
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
  late final String _viewType;

  @override
  void initState() {
    super.initState();
    _viewType = 'midi-player-${widget.midiUrl.hashCode}';
    ui_web.platformViewRegistry.registerViewFactory(_viewType, (int viewId) {
      final player = web.document.createElement('midi-player') as web.HTMLElement;
      player.setAttribute('src', widget.midiUrl);
      player.setAttribute('sound-font', '');
      player.style.width = '100%';
      player.style.height = '100%';
      return player;
    });
  }

  @override
  Widget build(BuildContext context) {
    return HtmlElementView(viewType: _viewType);
  }
}
