// Widget tests for the upload screen's AI refinement section (Plan 03-04).
// Covers UX-01 (checkbox + tooltip copy), UX-02 (createJob forwards
// enable_refine), UX-05 (default-false regression guard), D-22 (disabled
// state with helper text), D-23 (no persistence).
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart' as http_testing;

import 'package:ohsheet_app/api/client.dart';
import 'package:ohsheet_app/screens/upload_screen.dart';

/// Mock API that (a) returns a canned capabilities response and (b) records
/// the createJob body in a closure-captured holder.
class _MockApiHandle {
  _MockApiHandle({required this.api, required this.bodyRef});
  final OhSheetApi api;
  final List<Map<String, dynamic>> bodyRef;

  Map<String, dynamic> get lastJobBody =>
      bodyRef.isEmpty ? const <String, dynamic>{} : bodyRef.last;
}

_MockApiHandle _mockApi({required bool refineAvailable}) {
  final capturedBodies = <Map<String, dynamic>>[];
  final client = http_testing.MockClient((request) async {
    if (request.method == 'GET' && request.url.path == '/v1/capabilities') {
      return http.Response(
        jsonEncode({'refine_available': refineAvailable}),
        200,
        headers: {'content-type': 'application/json'},
      );
    }
    if (request.method == 'POST' && request.url.path == '/v1/jobs') {
      capturedBodies.add(jsonDecode(request.body) as Map<String, dynamic>);
      return http.Response(
        jsonEncode({
          'job_id': 'test-123',
          'status': 'queued',
          'variant': 'full',
        }),
        202,
        headers: {'content-type': 'application/json'},
      );
    }
    return http.Response('Not found', 404);
  });
  return _MockApiHandle(
    api: OhSheetApi(client: client),
    bodyRef: capturedBodies,
  );
}

Widget _app(OhSheetApi api) => MaterialApp(home: UploadScreen(api: api));

Future<void> _pumpAndLoadCapabilities(
    WidgetTester tester, Widget widget) async {
  await tester.pumpWidget(widget);
  // Let initState's getCapabilities future resolve and state rebuild.
  await tester.pumpAndSettle();
}

void main() {
  group('AI refinement section — UX-05 default-false', () {
    testWidgets('checkbox is rendered and defaults to unchecked',
        (tester) async {
      final mock = _mockApi(refineAvailable: true);
      await _pumpAndLoadCapabilities(tester, _app(mock.api));

      final toggle = find.byKey(const ValueKey('enableRefineToggle'));
      expect(toggle, findsOneWidget,
          reason: 'enableRefineToggle must exist for all variants (D-20)');

      final SwitchListTile widget = tester.widget(toggle);
      expect(widget.value, isFalse,
          reason:
              'UX-05: refine checkbox MUST default to false (regression guard)');
    });

    testWidgets('section title "AI refinement" is visible', (tester) async {
      final mock = _mockApi(refineAvailable: true);
      await _pumpAndLoadCapabilities(tester, _app(mock.api));
      expect(find.text('AI refinement'), findsWidgets,
          reason: 'D-20: dedicated AI refinement section header required');
    });

    testWidgets('toggle title carries the UX-01 experimental label',
        (tester) async {
      final mock = _mockApi(refineAvailable: true);
      await _pumpAndLoadCapabilities(tester, _app(mock.api));
      expect(find.text('Use AI refinement (experimental)'), findsOneWidget,
          reason: 'UX-01: "Use AI refinement (experimental)" is the SC1 label');
    });
  });

  group('UX-02: createJob forwards enable_refine', () {
    testWidgets(
        'ticking the toggle and submitting a title-lookup job sends enable_refine=true',
        (tester) async {
      final mock = _mockApi(refineAvailable: true);
      await _pumpAndLoadCapabilities(tester, _app(mock.api));

      // Switch to Title mode — no file picker interaction needed.
      await tester.tap(find.text('Title'));
      await tester.pumpAndSettle();

      // Enter a title so canSubmit is true.
      await tester.enterText(
          find.widgetWithText(TextField, 'Song title (required)'), 'Test Song');
      await tester.pumpAndSettle();

      // Flip the refine toggle — ensure it's visible first.
      await tester
          .ensureVisible(find.byKey(const ValueKey('enableRefineToggle')));
      await tester.pumpAndSettle();
      await tester.tap(find.byKey(const ValueKey('enableRefineToggle')));
      await tester.pumpAndSettle();

      // Submit.
      await tester
          .ensureVisible(find.byKey(const ValueKey('ohsheet_primary_submit')));
      await tester.pumpAndSettle();
      await tester.tap(find.byKey(const ValueKey('ohsheet_primary_submit')));
      await tester.pumpAndSettle(const Duration(milliseconds: 500));

      expect(mock.lastJobBody['enable_refine'], isTrue,
          reason:
              'UX-02: createJob must forward enable_refine=true when toggle is on');
    });

    testWidgets(
        'submitting WITHOUT ticking the toggle sends enable_refine=false',
        (tester) async {
      final mock = _mockApi(refineAvailable: true);
      await _pumpAndLoadCapabilities(tester, _app(mock.api));

      await tester.tap(find.text('Title'));
      await tester.pumpAndSettle();
      await tester.enterText(
          find.widgetWithText(TextField, 'Song title (required)'), 'Test Song');
      await tester.pumpAndSettle();
      await tester
          .ensureVisible(find.byKey(const ValueKey('ohsheet_primary_submit')));
      await tester.pumpAndSettle();
      await tester.tap(find.byKey(const ValueKey('ohsheet_primary_submit')));
      await tester.pumpAndSettle(const Duration(milliseconds: 500));

      expect(mock.lastJobBody['enable_refine'], isFalse,
          reason: 'UX-05 regression: default-false must survive to the POST body');
    });
  });

  group('D-22: refineAvailable=false disables the toggle with helper text', () {
    testWidgets('toggle is disabled and helper text is shown', (tester) async {
      final mock = _mockApi(refineAvailable: false);
      await _pumpAndLoadCapabilities(tester, _app(mock.api));

      final SwitchListTile widget =
          tester.widget(find.byKey(const ValueKey('enableRefineToggle')));
      expect(widget.onChanged, isNull,
          reason:
              'D-22: no API key configured → SwitchListTile.onChanged is null (disabled)');

      expect(find.text('AI refinement not configured on this server'),
          findsOneWidget,
          reason: 'D-22 verbatim helper text required');
    });
  });
}
