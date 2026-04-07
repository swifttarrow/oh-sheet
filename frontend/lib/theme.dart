/// Oh Sheet design system — kawaii / wireframe sticker palette.
library;

import 'package:flutter/material.dart';

abstract final class OhSheetColors {
  /// Primary teal (logo “Sheet!” tone).
  static const teal = Color(0xFF4DB6AC);
  static const tealBright = Color(0xFF6ECFC4);
  static const orange = Color(0xFFFFB300);
  static const cream = Color(0xFFFFF8F0);
  static const darkText = Color(0xFF2D3436);
  /// Chunky outline for sticker UI.
  static const inkStroke = Color(0xFF2D3436);
  static const mutedText = Color(0xFF636E72);
  static const error = Color(0xFFE17055);
  static const success = Color(0xFF00B894);
  static const pinkAccent = Color(0xFFFF80AB);
}

abstract final class OhSheetTheme {
  static ThemeData get light {
    final colorScheme = ColorScheme.fromSeed(
      seedColor: OhSheetColors.teal,
      primary: OhSheetColors.teal,
      secondary: OhSheetColors.orange,
      error: OhSheetColors.error,
      surface: Colors.white,
    );

    return ThemeData(
      useMaterial3: true,
      colorScheme: colorScheme,
      scaffoldBackgroundColor: OhSheetColors.cream,
      appBarTheme: const AppBarTheme(
        backgroundColor: OhSheetColors.cream,
        foregroundColor: OhSheetColors.darkText,
        elevation: 0,
        centerTitle: true,
        surfaceTintColor: Colors.transparent,
      ),
      filledButtonTheme: FilledButtonThemeData(
        style: FilledButton.styleFrom(
          backgroundColor: OhSheetColors.teal,
          foregroundColor: Colors.white,
          padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(26),
            side: const BorderSide(color: OhSheetColors.inkStroke, width: 2.5),
          ),
          textStyle: const TextStyle(fontSize: 16, fontWeight: FontWeight.w700),
        ),
      ),
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          foregroundColor: OhSheetColors.teal,
          side: const BorderSide(color: OhSheetColors.inkStroke, width: 2.5),
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
        ),
      ),
      segmentedButtonTheme: SegmentedButtonThemeData(
        style: ButtonStyle(
          backgroundColor: WidgetStateProperty.resolveWith((states) {
            if (states.contains(WidgetState.selected)) return OhSheetColors.teal;
            return Colors.white;
          }),
          foregroundColor: WidgetStateProperty.resolveWith((states) {
            if (states.contains(WidgetState.selected)) return Colors.white;
            return OhSheetColors.darkText;
          }),
          side: WidgetStateProperty.all(
            const BorderSide(color: OhSheetColors.inkStroke, width: 2),
          ),
          shape: WidgetStateProperty.all(
            RoundedRectangleBorder(borderRadius: BorderRadius.circular(22)),
          ),
          padding: WidgetStateProperty.all(
            const EdgeInsets.symmetric(horizontal: 12, vertical: 12),
          ),
        ),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: Colors.white,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(18),
          borderSide: const BorderSide(color: OhSheetColors.inkStroke, width: 2),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(18),
          borderSide: BorderSide(color: OhSheetColors.inkStroke.withValues(alpha: 0.35), width: 2),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(18),
          borderSide: const BorderSide(color: OhSheetColors.teal, width: 2.5),
        ),
        errorBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(18),
          borderSide: const BorderSide(color: OhSheetColors.error, width: 2),
        ),
        contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
      ),
      cardTheme: CardThemeData(
        elevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(22),
          side: const BorderSide(color: OhSheetColors.inkStroke, width: 2.5),
        ),
        color: Colors.white,
        shadowColor: OhSheetColors.inkStroke.withValues(alpha: 0.12),
      ),
      navigationBarTheme: NavigationBarThemeData(
        backgroundColor: Colors.white,
        elevation: 0,
        height: 68,
        indicatorColor: OhSheetColors.teal.withValues(alpha: 0.2),
        labelTextStyle: WidgetStateProperty.resolveWith((states) {
          if (states.contains(WidgetState.selected)) {
            return const TextStyle(
              color: OhSheetColors.teal,
              fontWeight: FontWeight.w800,
              fontSize: 12,
            );
          }
          return const TextStyle(color: OhSheetColors.mutedText, fontSize: 12, fontWeight: FontWeight.w600);
        }),
        iconTheme: WidgetStateProperty.resolveWith((states) {
          if (states.contains(WidgetState.selected)) {
            return const IconThemeData(color: OhSheetColors.teal, size: 26);
          }
          return const IconThemeData(color: OhSheetColors.mutedText, size: 24);
        }),
      ),
      navigationRailTheme: const NavigationRailThemeData(
        backgroundColor: Colors.white,
        indicatorColor: Color(0x334DB6AC),
        selectedIconTheme: IconThemeData(color: OhSheetColors.teal, size: 28),
        unselectedIconTheme: IconThemeData(color: OhSheetColors.mutedText, size: 26),
        selectedLabelTextStyle: TextStyle(
          color: OhSheetColors.teal,
          fontWeight: FontWeight.w800,
          fontSize: 13,
        ),
        unselectedLabelTextStyle: TextStyle(
          color: OhSheetColors.mutedText,
          fontWeight: FontWeight.w600,
          fontSize: 12,
        ),
      ),
    );
  }
}
