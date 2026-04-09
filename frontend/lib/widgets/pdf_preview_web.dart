/// Web-specific PDF preview using an iframe.
library;

import 'dart:ui_web' as ui_web;

import 'package:flutter/material.dart';
import 'package:web/web.dart' as web;

class WebPdfPreview extends StatefulWidget {
  const WebPdfPreview({super.key, required this.pdfUrl});
  final String pdfUrl;

  @override
  State<WebPdfPreview> createState() => _WebPdfPreviewState();
}

class _WebPdfPreviewState extends State<WebPdfPreview> {
  late final String _viewType;

  @override
  void initState() {
    super.initState();
    _viewType = 'pdf-preview-${widget.pdfUrl.hashCode}';
    ui_web.platformViewRegistry.registerViewFactory(_viewType, (int viewId) {
      final iframe = web.document.createElement('iframe') as web.HTMLIFrameElement;
      final separator = widget.pdfUrl.contains('?') ? '&' : '?';
      iframe.src = '${widget.pdfUrl}${separator}inline=true';
      iframe.style.width = '100%';
      iframe.style.height = '100%';
      iframe.style.border = 'none';
      return iframe;
    });
  }

  @override
  Widget build(BuildContext context) {
    return HtmlElementView(viewType: _viewType);
  }
}
