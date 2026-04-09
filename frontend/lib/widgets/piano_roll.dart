/// Platform-aware piano roll widget.
/// On web: custom canvas with note bars, Y-axis labels, and playback line.
/// On other platforms: fallback message.
library;

import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';

import '../theme.dart';
import 'piano_roll_stub.dart' if (dart.library.js_interop) 'piano_roll_web.dart';

class PianoRollWidget extends StatelessWidget {
  const PianoRollWidget({super.key, required this.midiUrl});
  final String midiUrl;

  @override
  Widget build(BuildContext context) {
    if (kIsWeb) {
      return WebPianoRoll(midiUrl: midiUrl);
    }
    return const Center(
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.music_note, color: OhSheetColors.teal),
          SizedBox(width: 8),
          Text(
            'Piano roll available in browser',
            style: TextStyle(color: OhSheetColors.mutedText, fontSize: 13),
          ),
        ],
      ),
    );
  }
}
