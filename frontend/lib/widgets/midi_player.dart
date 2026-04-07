/// Platform-aware MIDI player widget.
/// On Flutter Web: uses html-midi-player web component via HtmlElementView.
/// On other platforms: shows a fallback message.
library;

import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';

import '../theme.dart';
import 'midi_player_stub.dart' if (dart.library.js_interop) 'midi_player_web.dart';

class MidiPlayerWidget extends StatelessWidget {
  const MidiPlayerWidget({super.key, this.midiUrl});
  final String? midiUrl;

  @override
  Widget build(BuildContext context) {
    if (kIsWeb && midiUrl != null) {
      return WebMidiPlayer(midiUrl: midiUrl!);
    }
    return const Center(
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.music_note, color: OhSheetColors.teal),
          SizedBox(width: 8),
          Text(
            'Download MIDI to listen',
            style: TextStyle(color: OhSheetColors.mutedText, fontSize: 13),
          ),
        ],
      ),
    );
  }
}
