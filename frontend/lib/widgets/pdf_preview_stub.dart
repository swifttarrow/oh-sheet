/// Stub for non-web platforms.
library;

import 'package:flutter/material.dart';

class WebPdfPreview extends StatelessWidget {
  const WebPdfPreview({super.key, required this.pdfUrl});
  final String pdfUrl;

  @override
  Widget build(BuildContext context) {
    return const SizedBox.shrink();
  }
}
