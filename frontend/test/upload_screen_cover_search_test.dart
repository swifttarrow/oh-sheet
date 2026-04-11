// TDD: Tests for the "Find a clean piano cover" toggle on UploadScreen.
//
// This toggle is the frontend half of the cover_search feature. When on,
// the UploadScreen passes prefer_clean_source=true in the POST /v1/jobs
// body. The backend (IngestService) then probes the YouTube URL for
// metadata and searches for a clean piano cover to transcribe instead.
//
// The toggle only appears in YouTube mode — it's meaningless for audio
// uploads (user already picked their source) and title-only lookups
// (no YouTube URL to swap).
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart' as http_testing;

import 'package:ohsheet_app/api/client.dart';
import 'package:ohsheet_app/screens/upload_screen.dart';

const _youtubeKey = ValueKey('ohsheet_prefer_clean_source_toggle');

OhSheetApi _mockApi(void Function(Map<String, dynamic> body)? onCreateJob) {
  final mockClient = http_testing.MockClient((request) async {
    if (request.url.path == '/v1/jobs') {
      final body = jsonDecode(request.body) as Map<String, dynamic>;
      onCreateJob?.call(body);
      return http.Response(
        jsonEncode({
          'job_id': 'cs-test',
          'status': 'queued',
          'variant': 'full',
          'title': body['title'],
        }),
        202,
        headers: {'content-type': 'application/json'},
      );
    }
    return http.Response('Not found', 404);
  });
  return OhSheetApi(client: mockClient);
}

Widget _app(OhSheetApi api) => MaterialApp(home: UploadScreen(api: api));

Future<void> _selectYoutubeMode(WidgetTester tester) async {
  await tester.tap(find.text('YouTube'));
  await tester.pumpAndSettle();
}

Future<void> _enterYoutubeUrl(WidgetTester tester, String url) async {
  await tester.enterText(find.widgetWithText(TextField, 'YouTube URL'), url);
  await tester.pumpAndSettle();
}

Future<void> _submit(WidgetTester tester) async {
  await tester.ensureVisible(find.byKey(const ValueKey('ohsheet_primary_submit')));
  await tester.tap(find.byKey(const ValueKey('ohsheet_primary_submit')));
  await tester.pumpAndSettle();
}

void main() {
  group('Clean-source toggle visibility', () {
    testWidgets('toggle is visible in YouTube mode', (tester) async {
      await tester.pumpWidget(_app(_mockApi(null)));
      await _selectYoutubeMode(tester);
      expect(find.byKey(_youtubeKey), findsOneWidget);
    });

    testWidgets('toggle is NOT visible in audio mode', (tester) async {
      await tester.pumpWidget(_app(_mockApi(null)));
      // Default mode is audio — confirm toggle doesn't appear.
      expect(find.byKey(_youtubeKey), findsNothing);
    });

    testWidgets('toggle is NOT visible in MIDI mode', (tester) async {
      await tester.pumpWidget(_app(_mockApi(null)));
      await tester.tap(find.text('MIDI'));
      await tester.pumpAndSettle();
      expect(find.byKey(_youtubeKey), findsNothing);
    });

    testWidgets('toggle is NOT visible in title mode', (tester) async {
      await tester.pumpWidget(_app(_mockApi(null)));
      await tester.tap(find.text('Title'));
      await tester.pumpAndSettle();
      expect(find.byKey(_youtubeKey), findsNothing);
    });
  });

  group('Clean-source toggle submission', () {
    testWidgets('toggle defaults to OFF and submits prefer_clean_source=false',
        (tester) async {
      Map<String, dynamic>? captured;
      await tester.pumpWidget(_app(_mockApi((body) => captured = body)));
      await _selectYoutubeMode(tester);
      await _enterYoutubeUrl(tester, 'https://youtu.be/dQw4w9WgXcQ');
      await _submit(tester);

      expect(captured, isNotNull);
      // Default off — explicit false in body so backend sees the opt-in clearly.
      expect(captured!['prefer_clean_source'], false);
    });

    testWidgets('flipping toggle ON submits prefer_clean_source=true',
        (tester) async {
      Map<String, dynamic>? captured;
      await tester.pumpWidget(_app(_mockApi((body) => captured = body)));
      await _selectYoutubeMode(tester);
      await _enterYoutubeUrl(tester, 'https://youtu.be/dQw4w9WgXcQ');

      // Flip the toggle on. The SwitchListTile is identified by key.
      // Scroll into view first — the toggle sits below the artist field
      // and the test viewport may not show it by default.
      await tester.ensureVisible(find.byKey(_youtubeKey));
      await tester.pumpAndSettle();
      await tester.tap(find.byKey(_youtubeKey));
      await tester.pumpAndSettle();

      await _submit(tester);

      expect(captured, isNotNull);
      expect(captured!['prefer_clean_source'], true);
    });

    testWidgets(
        'switching to YouTube shows an explainer mentioning piano cover',
        (tester) async {
      await tester.pumpWidget(_app(_mockApi(null)));
      await _selectYoutubeMode(tester);
      // Any prose describing what the toggle does — the exact copy is
      // allowed to change but "piano cover" should anchor the explanation.
      expect(find.textContaining('piano cover'), findsAtLeastNWidgets(1));
    });
  });
}
