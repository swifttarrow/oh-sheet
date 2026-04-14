// Unit tests for OhSheetApi — enable_refine serialization + getCapabilities.
// Plan 03-03. Uses http_testing.MockClient to intercept HTTP calls.
import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart' as http_testing;

import 'package:ohsheet_app/api/client.dart';
import 'package:ohsheet_app/api/models.dart';

/// Builds an OhSheetApi with a MockClient that records the last createJob body
/// and returns a canned 202 JobSummary.
({OhSheetApi api, Map<String, dynamic> Function() lastBody}) _createJobMock() {
  Map<String, dynamic>? captured;
  final client = http_testing.MockClient((request) async {
    if (request.method == 'POST' && request.url.path == '/v1/jobs') {
      captured = jsonDecode(request.body) as Map<String, dynamic>;
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
  return (api: OhSheetApi(client: client), lastBody: () => captured ?? {});
}

OhSheetApi _capabilitiesMock({required bool refineAvailable, int statusCode = 200}) {
  final client = http_testing.MockClient((request) async {
    if (request.method == 'GET' && request.url.path == '/v1/capabilities') {
      if (statusCode != 200) {
        return http.Response('server error', statusCode);
      }
      return http.Response(
        jsonEncode({'refine_available': refineAvailable}),
        200,
        headers: {'content-type': 'application/json'},
      );
    }
    return http.Response('Not found', 404);
  });
  return OhSheetApi(client: client);
}

void main() {
  group('createJob enable_refine forwarding (UX-02)', () {
    test('enableRefine: true → body contains "enable_refine": true', () async {
      final mock = _createJobMock();
      await mock.api.createJob(title: 'x', enableRefine: true);
      expect(mock.lastBody()['enable_refine'], isTrue);
    });

    test('enableRefine: false → body contains "enable_refine": false', () async {
      final mock = _createJobMock();
      await mock.api.createJob(title: 'x', enableRefine: false);
      expect(mock.lastBody()['enable_refine'], isFalse);
    });

    test('default (no enableRefine arg) → body contains "enable_refine": false', () async {
      final mock = _createJobMock();
      await mock.api.createJob(title: 'x');
      expect(mock.lastBody().containsKey('enable_refine'), isTrue,
          reason: 'enable_refine must always be present so the backend can '
              'default unambiguously; do NOT make it conditional.');
      expect(mock.lastBody()['enable_refine'], isFalse);
    });
  });

  group('getCapabilities (D-22)', () {
    test('refine_available=true → Capabilities.refineAvailable == true', () async {
      final api = _capabilitiesMock(refineAvailable: true);
      final Capabilities caps = await api.getCapabilities();
      expect(caps.refineAvailable, isTrue);
    });

    test('refine_available=false → Capabilities.refineAvailable == false', () async {
      final api = _capabilitiesMock(refineAvailable: false);
      final Capabilities caps = await api.getCapabilities();
      expect(caps.refineAvailable, isFalse);
    });

    test('non-200 response throws ApiException', () async {
      final api = _capabilitiesMock(refineAvailable: false, statusCode: 500);
      expect(() => api.getCapabilities(), throwsA(isA<ApiException>()));
    });
  });
}
