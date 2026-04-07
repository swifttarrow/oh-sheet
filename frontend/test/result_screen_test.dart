/// TDD: Tests for the restyled result screen with mascot and download buttons.
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/testing.dart' as http_testing;
import 'package:http/http.dart' as http;

import 'package:ohsheet_app/api/client.dart';
import 'package:ohsheet_app/api/models.dart';
import 'package:ohsheet_app/screens/result_screen.dart';

JobSummary _fakeJob() => JobSummary(
      jobId: 'test-abc',
      status: 'succeeded',
      variant: 'full',
      title: 'Never Gonna Give You Up',
      artist: 'Rick Astley',
      result: {
        'pdf_uri': 'file:///tmp/score.pdf',
        'musicxml_uri': 'file:///tmp/score.xml',
        'humanized_midi_uri': 'file:///tmp/score.mid',
      },
    );

OhSheetApi _mockApi() => OhSheetApi(
      client: http_testing.MockClient((_) async => http.Response('{}', 404)),
    );

Widget _app() => MaterialApp(
      home: ResultScreen(api: _mockApi(), job: _fakeJob()),
    );

void main() {
  group('Result screen content', () {
    testWidgets('shows success mascot', (tester) async {
      await tester.pumpWidget(_app());
      final images = tester.widgetList<Image>(find.byType(Image));
      final paths = images
          .where((img) => img.image is AssetImage)
          .map((img) => (img.image as AssetImage).assetName);
      expect(paths, contains('assets/mascots/mascot-success.png'));
    });

    testWidgets('shows song title', (tester) async {
      await tester.pumpWidget(_app());
      expect(find.text('Never Gonna Give You Up'), findsOneWidget);
    });

    testWidgets('shows artist', (tester) async {
      await tester.pumpWidget(_app());
      expect(find.text('Rick Astley'), findsOneWidget);
    });

    testWidgets('shows PDF download button', (tester) async {
      await tester.pumpWidget(_app());
      expect(find.text('PDF'), findsOneWidget);
    });

    testWidgets('shows MIDI download button', (tester) async {
      await tester.pumpWidget(_app());
      // Button text is just "MIDI" (exact match), not the playback label
      expect(find.text('MIDI'), findsOneWidget);
    });

    testWidgets('shows MusicXML download button', (tester) async {
      await tester.pumpWidget(_app());
      expect(find.textContaining('MusicXML'), findsOneWidget);
    });

    testWidgets('shows transcribe another button', (tester) async {
      await tester.pumpWidget(_app());
      expect(find.textContaining('another'), findsOneWidget);
    });

    testWidgets('shows MIDI player section header', (tester) async {
      await tester.pumpWidget(_app());
      expect(find.text('Listen'), findsOneWidget);
    });
  });
}
