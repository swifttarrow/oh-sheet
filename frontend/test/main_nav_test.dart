/// TDD: Tests for the bottom navigation bar shell.
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/testing.dart' as http_testing;
import 'package:http/http.dart' as http;

import 'package:ohsheet_app/main.dart';

Widget _app() {
  return const OhSheetApp();
}

void main() {
  group('Bottom navigation bar', () {
    testWidgets('shows three tabs: Home, Library, Profile', (tester) async {
      await tester.pumpWidget(_app());
      expect(find.text('Home'), findsOneWidget);
      expect(find.text('Library'), findsOneWidget);
      expect(find.text('Profile'), findsOneWidget);
    });

    testWidgets('Home tab is selected by default', (tester) async {
      await tester.pumpWidget(_app());
      // The upload screen content should be visible by default
      expect(find.text('YouTube'), findsOneWidget); // from upload screen segments
    });

    testWidgets('tapping Library tab switches to library placeholder',
        (tester) async {
      await tester.pumpWidget(_app());
      await tester.tap(find.text('Library'));
      await tester.pumpAndSettle();
      expect(find.text('Community Library'), findsOneWidget);
    });

    testWidgets('tapping Profile tab switches to profile placeholder',
        (tester) async {
      await tester.pumpWidget(_app());
      await tester.tap(find.text('Profile'));
      await tester.pumpAndSettle();
      expect(find.text('Profile'), findsWidgets); // tab label + screen title
    });
  });
}
