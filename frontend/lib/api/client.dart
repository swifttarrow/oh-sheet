/// Thin HTTP client for the Oh Sheet pipeline API.
library;

import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;

import '../config.dart';
import 'models.dart';

class ApiException implements Exception {
  final int statusCode;
  final String body;
  ApiException(this.statusCode, this.body);
  @override
  String toString() => 'ApiException($statusCode): $body';
}

class OhSheetApi {
  OhSheetApi({http.Client? client}) : _client = client ?? http.Client();

  final http.Client _client;
  String get _base => AppConfig.apiBaseUrl;

  Uri _u(String path) => Uri.parse('$_base$path');

  // ---- uploads ---------------------------------------------------------

  Future<RemoteAudioFile> uploadAudio({
    required Uint8List bytes,
    required String filename,
  }) async {
    final req = http.MultipartRequest('POST', _u('/v1/uploads/audio'))
      ..files.add(http.MultipartFile.fromBytes('file', bytes, filename: filename));
    final streamed = await _client.send(req);
    final response = await http.Response.fromStream(streamed);
    if (response.statusCode != 200) {
      throw ApiException(response.statusCode, response.body);
    }
    return RemoteAudioFile.fromJson(jsonDecode(response.body) as Map<String, dynamic>);
  }

  Future<RemoteMidiFile> uploadMidi({
    required Uint8List bytes,
    required String filename,
  }) async {
    final req = http.MultipartRequest('POST', _u('/v1/uploads/midi'))
      ..files.add(http.MultipartFile.fromBytes('file', bytes, filename: filename));
    final streamed = await _client.send(req);
    final response = await http.Response.fromStream(streamed);
    if (response.statusCode != 200) {
      throw ApiException(response.statusCode, response.body);
    }
    return RemoteMidiFile.fromJson(jsonDecode(response.body) as Map<String, dynamic>);
  }

  // ---- jobs ------------------------------------------------------------

  /// Submit a job. Provide exactly one of ``audio``, ``midi``, or ``title``.
  ///
  /// ``preferCleanSource`` opts the user into the backend's clean-source
  /// search fast path: when true, the ingest stage looks for a piano or
  /// 8-bit cover of the song and transcribes that instead of the
  /// original YouTube URL. Pass ``null`` (the default) to omit the
  /// field entirely from the request body. Caller code for audio / MIDI
  /// / plain-title modes should not pass this parameter at all — it is
  /// meaningful only for YouTube URL submissions, and shipping it
  /// anyway is semantic noise in request logs (PR #47 review #4).
  Future<JobSummary> createJob({
    RemoteAudioFile? audio,
    RemoteMidiFile? midi,
    String? title,
    String? artist,
    bool skipHumanizer = false,
    bool? preferCleanSource,
  }) async {
    final body = <String, dynamic>{
      if (audio != null) 'audio': audio.toJson(),
      if (midi != null) 'midi': midi.toJson(),
      if (title != null && title.isNotEmpty) 'title': title,
      if (artist != null && artist.isNotEmpty) 'artist': artist,
      'skip_humanizer': skipHumanizer,
      if (preferCleanSource != null) 'prefer_clean_source': preferCleanSource,
    };
    final response = await _client.post(
      _u('/v1/jobs'),
      headers: {'content-type': 'application/json'},
      body: jsonEncode(body),
    );
    if (response.statusCode != 202) {
      throw ApiException(response.statusCode, response.body);
    }
    return JobSummary.fromJson(jsonDecode(response.body) as Map<String, dynamic>);
  }

  Future<JobSummary> getJob(String jobId) async {
    final response = await _client.get(_u('/v1/jobs/$jobId'));
    if (response.statusCode != 200) {
      throw ApiException(response.statusCode, response.body);
    }
    return JobSummary.fromJson(jsonDecode(response.body) as Map<String, dynamic>);
  }

  // ---- artifacts -------------------------------------------------------

  /// Public HTTP URL the OS can hand to a browser/download manager.
  /// Kind: 'pdf' | 'musicxml' | 'midi'.
  String artifactUrl(String jobId, String kind) =>
      '$_base/v1/artifacts/$jobId/$kind';

  void close() => _client.close();
}
