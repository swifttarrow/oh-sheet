/// TDD: Tests for the piano roll widget.
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/testing.dart' as http_testing;
import 'package:http/http.dart' as http;

import 'package:ohsheet_app/api/client.dart';
import 'package:ohsheet_app/api/models.dart';
import 'package:ohsheet_app/screens/result_screen.dart';
import 'package:ohsheet_app/widgets/piano_roll.dart';

JobSummary _fakeJob() => JobSummary(
      jobId: 'test-abc',
      status: 'succeeded',
      variant: 'full',
      title: 'Test Song',
      artist: 'Test Artist',
      result: {
        'pdf_uri': 'file:///tmp/score.pdf',
        'musicxml_uri': 'file:///tmp/score.xml',
        'humanized_midi_uri': 'file:///tmp/score.mid',
      },
    );

OhSheetApi _mockApi() => OhSheetApi(
      client: http_testing.MockClient((_) async => http.Response('{}', 404)),
    );

void main() {
  group('PianoRollWidget', () {
    testWidgets('renders without crashing', (tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: Scaffold(body: PianoRollWidget(midiUrl: 'http://test/midi')),
        ),
      );
      // On non-web test platform, should show fallback
      expect(find.textContaining('Piano roll'), findsOneWidget);
    });
  });

  group('Result screen has piano roll section', () {
    testWidgets('shows Listen section with piano roll', (tester) async {
      await tester.pumpWidget(
        MaterialApp(
          home: ResultScreen(api: _mockApi(), job: _fakeJob()),
        ),
      );
      await tester.pumpAndSettle();
      expect(find.text('Listen'), findsOneWidget);
    });
  });
}
