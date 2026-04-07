/// Stub for non-web platforms — WebMidiPlayer is never used but must exist
/// for the conditional import in midi_player.dart to compile.
library;

import 'package:flutter/material.dart';

class WebMidiPlayer extends StatelessWidget {
  const WebMidiPlayer({super.key, required this.midiUrl});
  final String midiUrl;

  @override
  Widget build(BuildContext context) {
    return const SizedBox.shrink();
  }
}
