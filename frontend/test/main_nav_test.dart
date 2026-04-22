// TDD: Tests for the bottom navigation bar shell.
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:ohsheet_app/main.dart';
import 'package:ohsheet_app/widgets/legal_disclaimer_dialog.dart';

Widget _app() {
  return const OhSheetApp();
}

Future<void> _pumpApp(WidgetTester tester) async {
  await tester.binding.setSurfaceSize(const Size(390, 844));
  addTearDown(() => tester.binding.setSurfaceSize(null));
  await tester.pumpWidget(_app());
  await tester.pumpAndSettle();
}

Future<void> _dismissLegalDisclaimer(WidgetTester tester) async {
  final continueButton = find.text('Continue responsibly');
  if (continueButton.evaluate().isEmpty) return;
  await tester.ensureVisible(continueButton);
  await tester.tap(continueButton);
  await tester.pumpAndSettle();
}

void main() {
  group('Bottom navigation bar', () {
    testWidgets('shows the legal disclaimer modal on first load', (tester) async {
      await _pumpApp(tester);
      expect(find.text(LegalDisclaimerDialog.titleText), findsOneWidget);
      expect(
        find.textContaining('Oh Sheet takes no responsibility'),
        findsOneWidget,
      );
    });

    testWidgets('shows three tabs: Home, Library, Profile', (tester) async {
      await _pumpApp(tester);
      await _dismissLegalDisclaimer(tester);
      expect(find.text('Home'), findsOneWidget);
      expect(find.text('Library'), findsOneWidget);
      expect(find.text('Profile'), findsOneWidget);
    });

    testWidgets('Home tab is selected by default', (tester) async {
      await _pumpApp(tester);
      await _dismissLegalDisclaimer(tester);
      // The upload screen content should be visible by default
      expect(find.text('YouTube'), findsOneWidget); // from upload screen segments
    });

    testWidgets('tapping Library tab switches to library placeholder',
        (tester) async {
      await _pumpApp(tester);
      await _dismissLegalDisclaimer(tester);
      await tester.tap(find.text('Library'));
      await tester.pumpAndSettle();
      expect(find.text('Community Library'), findsOneWidget);
    });

    testWidgets('tapping Profile tab switches to profile placeholder',
        (tester) async {
      await _pumpApp(tester);
      await _dismissLegalDisclaimer(tester);
      await tester.tap(find.text('Profile'));
      await tester.pumpAndSettle();
      expect(find.text('Profile'), findsWidgets); // tab label + screen title
    });
  });
}
