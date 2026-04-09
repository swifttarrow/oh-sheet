/// Stub for non-web platforms.
library;

import 'package:flutter/material.dart';

class WebSheetMusicViewer extends StatelessWidget {
  const WebSheetMusicViewer({
    super.key,
    required this.musicxmlUrl,
    required this.midiUrl,
  });
  final String musicxmlUrl;
  final String midiUrl;

  @override
  Widget build(BuildContext context) {
    return const SizedBox.shrink();
  }
}
