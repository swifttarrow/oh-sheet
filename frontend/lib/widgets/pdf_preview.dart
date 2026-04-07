/// Platform-aware PDF preview widget.
/// On Flutter Web: embeds an <iframe> pointing at the PDF URL.
/// On other platforms: shows a fallback message.
library;

import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';

import '../theme.dart';
import 'pdf_preview_stub.dart' if (dart.library.js_interop) 'pdf_preview_web.dart';

class PdfPreviewWidget extends StatelessWidget {
  const PdfPreviewWidget({super.key, required this.pdfUrl});
  final String pdfUrl;

  @override
  Widget build(BuildContext context) {
    if (kIsWeb) {
      return WebPdfPreview(pdfUrl: pdfUrl);
    }
    return const Center(
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.picture_as_pdf, color: OhSheetColors.teal),
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
