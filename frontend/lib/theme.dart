/// Oh Sheet design system — kawaii color palette and component themes.
library;

import 'package:flutter/material.dart';

abstract final class OhSheetColors {
  static const teal = Color(0xFF2EC4B6);
  static const orange = Color(0xFFF5A623);
  static const cream = Color(0xFFFFF8F0);
  static const darkText = Color(0xFF2D3436);
  static const mutedText = Color(0xFF636E72);
  static const error = Color(0xFFE17055);
  static const success = Color(0xFF00B894);
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
        backgroundColor: Colors.white,
        foregroundColor: OhSheetColors.darkText,
        elevation: 0,
        centerTitle: true,
      ),
      filledButtonTheme: FilledButtonThemeData(
        style: FilledButton.styleFrom(
          backgroundColor: OhSheetColors.teal,
          foregroundColor: Colors.white,
          padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
          textStyle: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
        ),
      ),
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          foregroundColor: OhSheetColors.teal,
          side: const BorderSide(color: OhSheetColors.teal),
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
          shape: WidgetStateProperty.all(
            RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
          ),
        ),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: Colors.white,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: BorderSide(color: Colors.grey.shade300),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: BorderSide(color: Colors.grey.shade300),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: const BorderSide(color: OhSheetColors.teal, width: 2),
        ),
        contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
      ),
      cardTheme: CardThemeData(
        elevation: 0,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        color: Colors.white,
      ),
      navigationBarTheme: NavigationBarThemeData(
        backgroundColor: Colors.white,
        indicatorColor: OhSheetColors.teal.withValues(alpha: 0.15),
        labelTextStyle: WidgetStateProperty.resolveWith((states) {
          if (states.contains(WidgetState.selected)) {
            return const TextStyle(
              color: OhSheetColors.teal,
              fontWeight: FontWeight.w600,
              fontSize: 12,
            );
          }
          return const TextStyle(color: OhSheetColors.mutedText, fontSize: 12);
        }),
        iconTheme: WidgetStateProperty.resolveWith((states) {
          if (states.contains(WidgetState.selected)) {
            return const IconThemeData(color: OhSheetColors.teal);
          }
          return const IconThemeData(color: OhSheetColors.mutedText);
        }),
      ),
    );
  }
}
