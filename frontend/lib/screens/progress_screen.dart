/// Live progress screen — subscribes to /v1/jobs/{id}/ws and renders the
/// pipeline as a stage list with a progress bar. On terminal events, pushes
/// the result screen (or shows the failure inline).
library;

import 'dart:async';

import 'package:flutter/material.dart';

import '../api/client.dart';
import '../api/models.dart';
import '../api/ws.dart';
import 'result_screen.dart';

class ProgressScreen extends StatefulWidget {
  const ProgressScreen({super.key, required this.api, required this.jobId});
  final OhSheetApi api;
  final String jobId;

  @override
  State<ProgressScreen> createState() => _ProgressScreenState();
}

class _ProgressScreenState extends State<ProgressScreen> {
  JobEventStream? _stream;
  StreamSubscription<JobEvent>? _sub;

  final List<JobEvent> _events = [];
  final Set<String> _completedStages = {};
  String? _currentStage;
  String? _failureMessage;
  bool _navigated = false;

  @override
  void initState() {
    super.initState();
    _connect();
  }

  void _connect() {
    final stream = JobEventStream.connect(widget.jobId);
    _stream = stream;
    _sub = stream.events.listen(
      _onEvent,
      onError: (e) => setState(() => _failureMessage = e.toString()),
    );
  }

  void _onEvent(JobEvent event) {
    setState(() {
      _events.add(event);
      switch (event.type) {
        case 'stage_started':
          _currentStage = event.stage;
          break;
        case 'stage_completed':
          if (event.stage != null) _completedStages.add(event.stage!);
          break;
        case 'stage_failed':
          _failureMessage = event.message ?? 'Stage ${event.stage} failed';
          break;
        case 'job_failed':
          _failureMessage = event.message ?? 'Job failed';
          break;
        case 'job_succeeded':
          _onSucceeded();
          break;
      }
    });
  }

  Future<void> _onSucceeded() async {
    if (_navigated || !mounted) return;
    _navigated = true;
    try {
      final summary = await widget.api.getJob(widget.jobId);
      if (!mounted) return;
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(
          builder: (_) => ResultScreen(api: widget.api, job: summary),
        ),
      );
    } catch (e) {
      setState(() => _failureMessage = 'Job finished but fetch failed: $e');
    }
  }

  @override
  void dispose() {
    _sub?.cancel();
    _stream?.close();
    super.dispose();
  }

  double get _progress {
    // Prefer the latest server-emitted progress if any.
    for (final event in _events.reversed) {
      if (event.progress != null) return event.progress!.clamp(0.0, 1.0);
    }
    // Otherwise estimate from completed stages out of 5.
    return (_completedStages.length / kPipelineStages.length).clamp(0.0, 1.0);
  }

  @override
  Widget build(BuildContext context) {
    final failed = _failureMessage != null;

    return Scaffold(
      appBar: AppBar(title: Text('Job ${widget.jobId}')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            LinearProgressIndicator(value: failed ? null : _progress),
            const SizedBox(height: 8),
            Text(
              failed
                  ? 'Failed'
                  : (_currentStage == null
                      ? 'Waiting for pipeline…'
                      : 'Running: $_currentStage'),
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 16),
            Expanded(
              child: ListView(
                children: [
                  for (final stage in kPipelineStages)
                    _StageRow(
                      stage: stage,
                      done: _completedStages.contains(stage),
                      active: _currentStage == stage &&
                          !_completedStages.contains(stage),
                    ),
                  const Divider(height: 32),
                  ..._events.reversed.map(
                    (e) => ListTile(
                      dense: true,
                      leading: _eventIcon(e.type),
                      title: Text('${e.type}${e.stage != null ? ' · ${e.stage}' : ''}'),
                      subtitle: e.message == null ? null : Text(e.message!),
                    ),
                  ),
                ],
              ),
            ),
            if (failed) ...[
              const SizedBox(height: 8),
              Text(
                _failureMessage!,
                style: TextStyle(color: Theme.of(context).colorScheme.error),
              ),
              const SizedBox(height: 8),
              FilledButton(
                onPressed: () => Navigator.of(context).pop(),
                child: const Text('Back'),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Icon _eventIcon(String type) {
    switch (type) {
      case 'job_succeeded':
      case 'stage_completed':
        return const Icon(Icons.check_circle, color: Colors.green);
      case 'job_failed':
      case 'stage_failed':
        return const Icon(Icons.error, color: Colors.red);
      case 'stage_started':
        return const Icon(Icons.play_arrow);
      default:
        return const Icon(Icons.info_outline);
    }
  }
}

class _StageRow extends StatelessWidget {
  const _StageRow({required this.stage, required this.done, required this.active});
  final String stage;
  final bool done;
  final bool active;

  @override
  Widget build(BuildContext context) {
    final IconData icon;
    final Color color;
    if (done) {
      icon = Icons.check_circle;
      color = Colors.green;
    } else if (active) {
      icon = Icons.autorenew;
      color = Theme.of(context).colorScheme.primary;
    } else {
      icon = Icons.radio_button_unchecked;
      color = Theme.of(context).disabledColor;
    }
    return ListTile(
      leading: Icon(icon, color: color),
      title: Text(stage),
      trailing: active
          ? const SizedBox(
              width: 16,
              height: 16,
              child: CircularProgressIndicator(strokeWidth: 2),
            )
          : null,
    );
  }
}
