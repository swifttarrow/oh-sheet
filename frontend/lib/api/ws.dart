/// WebSocket subscription to /v1/jobs/{job_id}/ws.
///
/// Late subscribers get a replay of all events that have already happened —
/// the backend handles that for us, so we just open the channel and forward
/// JSON frames as JobEvents.
library;

import 'dart:async';
import 'dart:convert';

import 'package:web_socket_channel/web_socket_channel.dart';

import '../config.dart';
import 'models.dart';

class JobEventStream {
  JobEventStream._(this._channel, this._controller);

  final WebSocketChannel _channel;
  final StreamController<JobEvent> _controller;
  StreamSubscription? _sub;

  static JobEventStream connect(String jobId) {
    final url = '${AppConfig.wsBaseUrl}/v1/jobs/$jobId/ws';
    final channel = WebSocketChannel.connect(Uri.parse(url));
    final controller = StreamController<JobEvent>.broadcast();
    final stream = JobEventStream._(channel, controller);
    stream._wire();
    return stream;
  }

  void _wire() {
    _sub = _channel.stream.listen(
      (raw) {
        try {
          final json = jsonDecode(raw as String) as Map<String, dynamic>;
          if (json.containsKey('error')) {
            _controller.addError(StateError(json['error'] as String));
            return;
          }
          final event = JobEvent.fromJson(json);
          _controller.add(event);
          if (event.isTerminal) {
            _controller.close();
          }
        } catch (e, st) {
          _controller.addError(e, st);
        }
      },
      onError: _controller.addError,
      onDone: () {
        if (!_controller.isClosed) _controller.close();
      },
    );
  }

  Stream<JobEvent> get events => _controller.stream;

  Future<void> close() async {
    await _sub?.cancel();
    await _channel.sink.close();
    if (!_controller.isClosed) await _controller.close();
  }
}
