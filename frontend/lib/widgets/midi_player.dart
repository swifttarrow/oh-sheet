/// Platform-aware MIDI player widget.
/// On Flutter Web: uses html-midi-player web component via HtmlElementView.
/// On other platforms: shows a placeholder with download prompt.
library;

import 'package:flutter/material.dart';

import '../theme.dart';

/// Placeholder widget used in tests and non-web platforms.
/// On web, this is replaced by the HtmlElementView in midi_player_web.dart.
class MidiPlayerWidget extends StatelessWidget {
  const MidiPlayerWidget({super.key, this.midiUrl});
  final String? midiUrl;

  @override
  Widget build(BuildContext context) {
    return const Center(
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.music_note, color: OhSheetColors.teal),
          SizedBox(width: 8),
          Text(
            'MIDI playback available in browser',
            style: TextStyle(color: OhSheetColors.mutedText, fontSize: 13),
          ),
        ],
      ),
    );
  }
}
