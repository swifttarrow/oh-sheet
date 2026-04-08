// TDD: Tests for the Oh Sheet kawaii theme system.
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:ohsheet_app/theme.dart';

void main() {
  group('OhSheetTheme', () {
    test('provides a light ThemeData', () {
      final theme = OhSheetTheme.light;
      expect(theme, isA<ThemeData>());
      expect(theme.useMaterial3, isTrue);
    });

    test('primary color is teal', () {
      final theme = OhSheetTheme.light;
      expect(theme.colorScheme.primary, OhSheetColors.teal);
    });

    test('scaffold background is cream', () {
      final theme = OhSheetTheme.light;
      expect(theme.scaffoldBackgroundColor, OhSheetColors.cream);
    });

    test('filled buttons have rounded pill shape', () {
      final theme = OhSheetTheme.light;
      final buttonTheme = theme.filledButtonTheme.style!;
      final shape = buttonTheme.shape!.resolve({});
      expect(shape, isA<RoundedRectangleBorder>());
      final rrb = shape as RoundedRectangleBorder;
      expect(rrb.borderRadius, BorderRadius.circular(24));
    });

    test('cards have rounded corners', () {
      final theme = OhSheetTheme.light;
      final cardTheme = theme.cardTheme;
      final shape = cardTheme.shape as RoundedRectangleBorder;
      expect(shape.borderRadius, BorderRadius.circular(16));
    });
  });

  group('OhSheetColors', () {
    test('defines all required palette colors', () {
      expect(OhSheetColors.teal, isA<Color>());
      expect(OhSheetColors.orange, isA<Color>());
      expect(OhSheetColors.cream, isA<Color>());
      expect(OhSheetColors.darkText, isA<Color>());
      expect(OhSheetColors.mutedText, isA<Color>());
      expect(OhSheetColors.error, isA<Color>());
      expect(OhSheetColors.success, isA<Color>());
    });
  });
}
