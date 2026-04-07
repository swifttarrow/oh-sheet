// TDD: Tests for YouTube URL input mode on UploadScreen.
// Written FIRST, before implementation.
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart' as http_testing;
import 'dart:convert';

import 'package:ohsheet_app/api/client.dart';
import 'package:ohsheet_app/screens/upload_screen.dart';

/// Mock API that returns a canned JobSummary for any createJob call.
OhSheetApi _mockApi() {
  final mockClient = http_testing.MockClient((request) async {
    if (request.url.path == '/v1/jobs') {
      return http.Response(
        jsonEncode({
          'job_id': 'test-123',
          'status': 'queued',
          'variant': 'full',
          'title': 'https://youtube.com/watch?v=dQw4w9WgXcQ',
        }),
        202,
        headers: {'content-type': 'application/json'},
      );
    }
    return http.Response('Not found', 404);
  });
  return OhSheetApi(client: mockClient);
}

Widget _app(OhSheetApi api) => MaterialApp(
      home: UploadScreen(api: api),
    );

void main() {
  group('YouTube segment button', () {
    testWidgets('YouTube segment is visible in the SegmentedButton',
        (tester) async {
      await tester.pumpWidget(_app(_mockApi()));
      // There should be a "YouTube" segment label
      expect(find.text('YouTube'), findsOneWidget);
    });

    testWidgets('selecting YouTube mode shows URL text field', (tester) async {
      await tester.pumpWidget(_app(_mockApi()));
      // Tap the YouTube segment
      await tester.tap(find.text('YouTube'));
      await tester.pumpAndSettle();

      // Should show a URL input field with YouTube-specific hint
      expect(find.widgetWithText(TextField, 'YouTube URL'), findsOneWidget);
    });

    testWidgets('selecting YouTube mode hides the file picker', (tester) async {
      await tester.pumpWidget(_app(_mockApi()));
      await tester.tap(find.text('YouTube'));
      await tester.pumpAndSettle();

      // File picker button should NOT be visible in YouTube mode
      expect(find.text('Pick audio file (mp3/wav/flac/m4a)'), findsNothing);
      expect(find.text('Pick MIDI file (.mid/.midi)'), findsNothing);
    });
  });

  group('YouTube URL validation', () {
    testWidgets('submit button is disabled when URL field is empty',
        (tester) async {
      await tester.pumpWidget(_app(_mockApi()));
      await tester.tap(find.text('YouTube'));
      await tester.pumpAndSettle();

      // Transcribe button should exist but be disabled
      final button = tester.widget<FilledButton>(find.byType(FilledButton));
      expect(button.onPressed, isNull);
    });

    testWidgets('submit button is enabled with a valid YouTube URL',
        (tester) async {
      await tester.pumpWidget(_app(_mockApi()));
      await tester.tap(find.text('YouTube'));
      await tester.pumpAndSettle();

      // Type a valid YouTube URL
      await tester.enterText(
        find.widgetWithText(TextField, 'YouTube URL'),
        'https://youtube.com/watch?v=dQw4w9WgXcQ',
      );
      await tester.pumpAndSettle();

      // Transcribe button should be enabled
      final button = tester.widget<FilledButton>(find.byType(FilledButton));
      expect(button.onPressed, isNotNull);
    });

    testWidgets('shows error for invalid URL format', (tester) async {
      await tester.pumpWidget(_app(_mockApi()));
      await tester.tap(find.text('YouTube'));
      await tester.pumpAndSettle();

      // Type an invalid URL
      await tester.enterText(
        find.widgetWithText(TextField, 'YouTube URL'),
        'not-a-youtube-url',
      );
      await tester.pumpAndSettle();

      // Should show validation error
      expect(find.textContaining('valid YouTube URL'), findsOneWidget);
    });
  });

  group('YouTube job submission', () {
    testWidgets('submitting a YouTube URL sends title field to API',
        (tester) async {
      String? capturedTitle;
      final mockClient = http_testing.MockClient((request) async {
        if (request.url.path == '/v1/jobs') {
          final body = jsonDecode(request.body) as Map<String, dynamic>;
          capturedTitle = body['title'] as String?;
          return http.Response(
            jsonEncode({
              'job_id': 'yt-test-456',
              'status': 'queued',
              'variant': 'full',
              'title': capturedTitle,
            }),
            202,
            headers: {'content-type': 'application/json'},
          );
        }
        return http.Response('Not found', 404);
      });
      final api = OhSheetApi(client: mockClient);

      await tester.pumpWidget(_app(api));
      await tester.tap(find.text('YouTube'));
      await tester.pumpAndSettle();

      await tester.enterText(
        find.widgetWithText(TextField, 'YouTube URL'),
        'https://youtube.com/watch?v=dQw4w9WgXcQ',
      );
      await tester.pumpAndSettle();

      // Tap Transcribe
      await tester.tap(find.text('Transcribe'));
      await tester.pumpAndSettle();

      // The API should have received the YouTube URL as the title
      expect(capturedTitle, 'https://youtube.com/watch?v=dQw4w9WgXcQ');
    });
  });
}
