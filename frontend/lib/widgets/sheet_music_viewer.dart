/// Platform-aware interactive sheet music viewer.
/// On web: renders MusicXML via OSMD with playback cursor.
/// On other platforms: falls back to PDF download prompt.
library;

import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';

import '../theme.dart';
import 'sheet_music_viewer_stub.dart'
    if (dart.library.js_interop) 'sheet_music_viewer_web.dart';

class SheetMusicViewer extends StatelessWidget {
  const SheetMusicViewer({
    super.key,
    required this.musicxmlUrl,
    required this.midiUrl,
  });
  final String musicxmlUrl;
  final String midiUrl;

  @override
  Widget build(BuildContext context) {
    if (kIsWeb) {
      return WebSheetMusicViewer(musicxmlUrl: musicxmlUrl, midiUrl: midiUrl);
    }
    return const Center(
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.music_note, color: OhSheetColors.teal),
          SizedBox(width: 8),
          Text(
            'Download PDF to view sheet music',
            style: TextStyle(color: OhSheetColors.mutedText, fontSize: 13),
          ),
        ],
      ),
    );
  }
}
