/// Breakpoints and layout helpers for phone, tablet, and web/desktop.
library;

import 'package:flutter/material.dart';

/// Layout breakpoints (logical pixels, width).
abstract final class OhSheetBreakpoints {
  /// Side navigation instead of bottom bar.
  static const double sideNav = 720;

  /// Two-column result layout (preview + actions).
  static const double resultTwoColumn = 900;

  /// Form / flow content max width on large canvases.
  static const double contentNarrow = 520;

  /// Wider flows (progress, mixed content).
  static const double contentMedium = 640;

  /// Result page max width.
  static const double contentWide = 1040;
}

extension OhSheetLayoutContext on BuildContext {
  double get ohSheetWidth => MediaQuery.sizeOf(this).width;

  bool get ohSheetUseSideNav =>
      ohSheetWidth >= OhSheetBreakpoints.sideNav;

  bool get ohSheetResultTwoColumn =>
      ohSheetWidth >= OhSheetBreakpoints.resultTwoColumn;
}

/// Centers [child] and caps width so forms do not stretch edge-to-edge on web.
class OhSheetResponsiveBody extends StatelessWidget {
  const OhSheetResponsiveBody({
    super.key,
    required this.child,
    this.maxWidth = OhSheetBreakpoints.contentNarrow,
    this.padding = const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
    this.alignTop = true,
  });

  final Widget child;
  final double maxWidth;
  final EdgeInsets padding;
  final bool alignTop;

  @override
  Widget build(BuildContext context) {
    return Align(
      alignment: alignTop ? Alignment.topCenter : Alignment.center,
      child: ConstrainedBox(
        constraints: BoxConstraints(maxWidth: maxWidth),
        child: Padding(padding: padding, child: child),
      ),
    );
  }
}
