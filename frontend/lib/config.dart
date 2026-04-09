/// Runtime configuration.
///
/// Override at build time with:
///   flutter run --dart-define=API_BASE_URL=http://192.168.1.42:8000
///
/// Defaults pick a sensible host per platform: Android emulators reach the
/// host machine at 10.0.2.2; everything else uses localhost.
library;

import 'dart:io' show Platform;

import 'package:flutter/foundation.dart';

class AppConfig {
  static const String _envBaseUrl = String.fromEnvironment('API_BASE_URL');

  static String get apiBaseUrl {
    if (_envBaseUrl.isNotEmpty) return _envBaseUrl;
    if (kIsWeb) {
      // In production (Cloud Run), use same-origin relative URLs.
      // In local dev, the page is served on a different port, so use localhost:8000.
      final host = Uri.base.host;
      if (host == 'localhost' || host == '127.0.0.1') {
        return 'http://localhost:8000';
      }
      return '';
    }
    try {
      if (Platform.isAndroid) return 'http://10.0.2.2:8000';
    } catch (_) {
      // Platform isn't available on web; fall through.
    }
    return 'http://localhost:8000';
  }

  /// ws:// (or wss://) variant of the API base, used for the live job stream.
  static String get wsBaseUrl {
    final base = apiBaseUrl;
    if (base.startsWith('https://')) return 'wss://${base.substring(8)}';
    if (base.startsWith('http://')) return 'ws://${base.substring(7)}';
    // When apiBaseUrl is empty (same-origin production), construct absolute
    // WebSocket URL from the page origin — relative URLs don't work for WS.
    if (base.isEmpty && kIsWeb) {
      final scheme = Uri.base.scheme == 'https' ? 'wss' : 'ws';
      return '$scheme://${Uri.base.host}:${Uri.base.port}';
    }
    return base;
  }
}
