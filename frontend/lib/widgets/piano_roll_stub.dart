/// Stub for non-web platforms.
library;

import 'package:flutter/material.dart';

class WebPianoRoll extends StatelessWidget {
  const WebPianoRoll({super.key, required this.midiUrl});
  final String midiUrl;

  @override
  Widget build(BuildContext context) {
    return const SizedBox.shrink();
  }
}
