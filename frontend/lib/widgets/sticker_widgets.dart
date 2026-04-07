/// Die-cut “sticker” surfaces and playful controls matching wireframe style.
library;

import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../theme.dart';

/// Outer stroke for sticker panels and chunky controls.
abstract final class OhSheetStickerStyle {
  static const double radius = 24;
  static const double radiusLg = 28;
  static const double borderWidth = 3;
  static const double shadowDx = 3;
  static const double shadowDy = 4;

  static List<BoxShadow> get stickerShadows => [
        BoxShadow(
          color: OhSheetColors.inkStroke.withValues(alpha: 0.12),
          offset: const Offset(shadowDx, shadowDy),
          blurRadius: 0,
        ),
        BoxShadow(
          color: OhSheetColors.inkStroke.withValues(alpha: 0.08),
          offset: const Offset(0, 6),
          blurRadius: 18,
        ),
      ];
}

/// White panel with thick outline + sticker shadow (wireframe cards).
class OhSheetSticker extends StatelessWidget {
  const OhSheetSticker({
    super.key,
    required this.child,
    this.padding = const EdgeInsets.all(20),
    this.backgroundColor = Colors.white,
    this.radius = OhSheetStickerStyle.radius,
  });

  final Widget child;
  final EdgeInsetsGeometry padding;
  final Color backgroundColor;
  final double radius;

  @override
  Widget build(BuildContext context) {
    return DecoratedBox(
      decoration: BoxDecoration(
        color: backgroundColor,
        borderRadius: BorderRadius.circular(radius),
        border: Border.all(
          color: OhSheetColors.inkStroke,
          width: OhSheetStickerStyle.borderWidth,
        ),
        boxShadow: OhSheetStickerStyle.stickerShadows,
      ),
      child: Padding(padding: padding, child: child),
    );
  }
}

/// Dashed rounded frame (drop-zone look) with tappable interior.
class OhSheetDashedPickZone extends StatelessWidget {
  const OhSheetDashedPickZone({
    super.key,
    required this.onTap,
    required this.child,
    this.minHeight = 120,
  });

  final VoidCallback? onTap;
  final Widget child;
  final double minHeight;

  @override
  Widget build(BuildContext context) {
    return Material(
      color: OhSheetColors.teal.withValues(alpha: 0.08),
      borderRadius: BorderRadius.circular(OhSheetStickerStyle.radiusLg),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(OhSheetStickerStyle.radiusLg),
        splashColor: OhSheetColors.teal.withValues(alpha: 0.15),
        child: CustomPaint(
          painter: _DashedRRectPainter(
            color: OhSheetColors.teal,
            strokeWidth: 2.5,
            radius: OhSheetStickerStyle.radiusLg,
          ),
          child: ConstrainedBox(
            constraints: BoxConstraints(minHeight: minHeight),
            child: Center(child: child),
          ),
        ),
      ),
    );
  }
}

class _DashedRRectPainter extends CustomPainter {
  _DashedRRectPainter({
    required this.color,
    required this.strokeWidth,
    required this.radius,
  });

  final Color color;
  final double strokeWidth;
  final double radius;

  static const double _dashLength = 10;
  static const double _gapLength = 6;

  @override
  void paint(Canvas canvas, Size size) {
    final half = strokeWidth / 2;
    final rect = Rect.fromLTWH(half, half, size.width - strokeWidth, size.height - strokeWidth);
    final rrect = RRect.fromRectAndRadius(rect, Radius.circular(radius));
    final path = Path()..addRRect(rrect);

    final paint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = strokeWidth
      ..strokeCap = StrokeCap.round;

    for (final metric in path.computeMetrics()) {
      var d = 0.0;
      while (d < metric.length) {
        final end = math.min(d + _dashLength, metric.length);
        canvas.drawPath(metric.extractPath(d, end), paint);
        d = end + _gapLength;
      }
    }
  }

  @override
  bool shouldRepaint(covariant _DashedRRectPainter oldDelegate) =>
      oldDelegate.color != color ||
      oldDelegate.strokeWidth != strokeWidth ||
      oldDelegate.radius != radius;
}

/// Primary CTA: teal gradient, ink border, sticker shadow (logo energy).
class OhSheetStickerCTA extends StatelessWidget {
  const OhSheetStickerCTA({
    super.key,
    required this.onPressed,
    required this.label,
    this.icon,
    this.loading = false,
  });

  final VoidCallback? onPressed;
  final String label;
  final IconData? icon;
  final bool loading;

  @override
  Widget build(BuildContext context) {
    final enabled = onPressed != null && !loading;
    return DecoratedBox(
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(OhSheetStickerStyle.radiusLg),
        gradient: enabled
            ? const LinearGradient(
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
                colors: [
                  OhSheetColors.tealBright,
                  OhSheetColors.teal,
                ],
              )
            : LinearGradient(
                colors: [
                  Colors.grey.shade400,
                  Colors.grey.shade500,
                ],
              ),
        border: Border.all(
          color: OhSheetColors.inkStroke,
          width: OhSheetStickerStyle.borderWidth,
        ),
        boxShadow: enabled ? OhSheetStickerStyle.stickerShadows : null,
      ),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          onTap: enabled ? onPressed : null,
          borderRadius: BorderRadius.circular(OhSheetStickerStyle.radiusLg - 2),
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 16),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              mainAxisSize: MainAxisSize.min,
              children: [
                if (loading)
                  const SizedBox(
                    width: 22,
                    height: 22,
                    child: CircularProgressIndicator(
                      strokeWidth: 2.5,
                      color: Colors.white,
                    ),
                  )
                else if (icon != null) ...[
                  Icon(icon, color: Colors.white, size: 22),
                  const SizedBox(width: 10),
                ],
                Text(
                  label,
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 17,
                    fontWeight: FontWeight.w800,
                    letterSpacing: 0.3,
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

/// Section title with a playful color accent bar.
class OhSheetStickerSectionTitle extends StatelessWidget {
  const OhSheetStickerSectionTitle({super.key, required this.text, this.accent = OhSheetColors.pinkAccent});

  final String text;
  final Color accent;

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Container(
          width: 5,
          height: 22,
          decoration: BoxDecoration(
            color: accent,
            borderRadius: BorderRadius.circular(3),
            border: Border.all(color: OhSheetColors.inkStroke, width: 1.5),
          ),
        ),
        const SizedBox(width: 10),
        Text(
          text,
          style: const TextStyle(
            fontSize: 18,
            fontWeight: FontWeight.w800,
            color: OhSheetColors.darkText,
            letterSpacing: -0.2,
          ),
        ),
      ],
    );
  }
}

/// Sticker frame with inner clip for PDF/MIDI embeds (no inner padding).
class OhSheetStickerClip extends StatelessWidget {
  const OhSheetStickerClip({
    super.key,
    required this.child,
    this.height,
  });

  final Widget child;
  final double? height;

  @override
  Widget build(BuildContext context) {
    const r = OhSheetStickerStyle.radius;
    const bw = OhSheetStickerStyle.borderWidth;
    const inner = r - bw;
    Widget innerChild = child;
    if (height != null) {
      innerChild = SizedBox(height: height, width: double.infinity, child: child);
    }
    return DecoratedBox(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(r),
        border: Border.all(color: OhSheetColors.inkStroke, width: bw),
        boxShadow: OhSheetStickerStyle.stickerShadows,
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(inner > 0 ? inner : 0),
        child: innerChild,
      ),
    );
  }
}
