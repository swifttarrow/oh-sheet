// TDD: Tests for the welcome hero section on UploadScreen.
import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/testing.dart' as http_testing;
import 'package:http/http.dart' as http;

import 'package:ohsheet_app/api/client.dart';
import 'package:ohsheet_app/screens/upload_screen.dart';

Widget _app() => MaterialApp(
      home: UploadScreen(api: OhSheetApi(client: http_testing.MockClient(
        (_) async => http.Response('{}', 404),
      ))),
    );

void main() {
  group('Welcome hero section', () {
    testWidgets('displays mascot image', (tester) async {
      await tester.pumpWidget(_app());
      await tester.pumpAndSettle();
      expect(find.byType(SvgPicture), findsWidgets);
    });

    testWidgets('renders mascot without clipping it into a rounded tile', (
      tester,
    ) async {
      await tester.pumpWidget(_app());
      await tester.pumpAndSettle();

      final svg = tester.widget<SvgPicture>(find.byType(SvgPicture).first);
      expect(svg.clipBehavior, Clip.none);
      expect(svg.allowDrawingOutsideViewBox, isTrue);
    });

    testWidgets('displays headline text', (tester) async {
      await tester.pumpWidget(_app());
      expect(
        find.text('Turn any song into piano sheet music'),
        findsOneWidget,
      );
    });

    testWidgets('displays subtitle text', (tester) async {
      await tester.pumpWidget(_app());
      expect(
        find.textContaining('Upload audio'),
        findsOneWidget,
      );
    });

    testWidgets('upload form is still present below hero', (tester) async {
      await tester.pumpWidget(_app());
      // Segmented buttons should still exist
      expect(find.text('Audio'), findsOneWidget);
      expect(find.text('YouTube'), findsOneWidget);
    });

    testWidgets('transcribe button says Let\'s go!', (tester) async {
      await tester.pumpWidget(_app());
      expect(find.text("Let's go!"), findsOneWidget);
    });
  });
}
