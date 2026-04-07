/// Live progress screen — subscribes to /v1/jobs/{id}/ws and renders the
/// pipeline with stage-specific mascot images, sticker badges, and rotating tips.
library;

import 'dart:async';

import 'package:flutter/material.dart';

import '../api/client.dart';
import '../api/models.dart';
import '../api/ws.dart';
import '../theme.dart';
import 'result_screen.dart';

// ---------------------------------------------------------------------------
// Public helpers (tested independently)
// ---------------------------------------------------------------------------

String friendlyStageName(String stage) => switch (stage) {
      'ingest' => 'Preparing',
      'transcribe' => 'Transcribing',
      'arrange' => 'Arranging',
      'humanize' => 'Humanizing',
      'engrave' => 'Engraving',
      _ => stage,
    };

String mascotAssetForStage(String? stage) => switch (stage) {
      'ingest' => 'assets/mascots/mascot-progress-ingest.png',
      'transcribe' => 'assets/mascots/mascot-progress-transcribe.png',
      'arrange' || 'humanize' => 'assets/mascots/mascot-progress-arrange.png',
      'engrave' => 'assets/mascots/mascot-progress-engrave.png',
      _ => 'assets/mascots/mascot-progress-ingest.png',
    };

const pipelineTips = [
  'Most songs take 15–45 seconds.',
  'The AI is analyzing rhythm, melody, and harmony.',
  'Your piano arrangement will have right and left hand parts.',
  'Difficulty is rated automatically from 1–10.',
  'The final PDF is typeset with LilyPond — publication quality.',
];

// Display stages (hide humanize since the mascot reuses arrange)
const _displayStages = ['ingest', 'transcribe', 'arrange', 'engrave'];

// ---------------------------------------------------------------------------
// Screen
// ---------------------------------------------------------------------------

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
  int _tipIndex = 0;
  Timer? _tipTimer;

  @override
  void initState() {
    super.initState();
    _connect();
    _tipTimer = Timer.periodic(const Duration(seconds: 5), (_) {
      if (mounted) setState(() => _tipIndex = (_tipIndex + 1) % pipelineTips.length);
    });
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
        case 'stage_completed':
          if (event.stage != null) _completedStages.add(event.stage!);
        case 'stage_failed':
          _failureMessage = event.message ?? 'Stage ${event.stage} failed';
        case 'job_failed':
          _failureMessage = event.message ?? 'Job failed';
        case 'job_succeeded':
          _onSucceeded();
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
    _tipTimer?.cancel();
    _sub?.cancel();
    _stream?.close();
    super.dispose();
  }

  double get _progress {
    for (final event in _events.reversed) {
      if (event.progress != null) return event.progress!.clamp(0.0, 1.0);
    }
    return (_completedStages.length / kPipelineStages.length).clamp(0.0, 1.0);
  }

  @override
  Widget build(BuildContext context) {
    final failed = _failureMessage != null;

    return Scaffold(
      appBar: AppBar(
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () => Navigator.of(context).pop(),
        ),
        title: const Text('Oh Sheet'),
      ),
      body: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 24),
        child: Column(
          children: [
            const SizedBox(height: 16),

            // Mascot area
            AnimatedSwitcher(
              duration: const Duration(milliseconds: 400),
              child: Image.asset(
                failed
                    ? 'assets/mascots/mascot-error.png'
                    : mascotAssetForStage(_currentStage),
                key: ValueKey(failed ? 'error' : _currentStage),
                height: 180,
              ),
            ),
            const SizedBox(height: 20),

            // Stage badge row
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                for (final stage in _displayStages) ...[
                  _StageBadge(
                    label: friendlyStageName(stage),
                    done: _completedStages.contains(stage),
                    active: _currentStage == stage && !_completedStages.contains(stage),
                  ),
                  if (stage != _displayStages.last) const SizedBox(width: 8),
                ],
              ],
            ),
            const SizedBox(height: 24),

            // Progress bar
            ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: LinearProgressIndicator(
                value: failed ? null : _progress,
                minHeight: 10,
                backgroundColor: Colors.grey.shade200,
                valueColor: AlwaysStoppedAnimation(
                  failed ? OhSheetColors.error : OhSheetColors.teal,
                ),
              ),
            ),
            const SizedBox(height: 24),

            // Rotating tip
            if (!failed)
              AnimatedSwitcher(
                duration: const Duration(milliseconds: 300),
                child: Text(
                  pipelineTips[_tipIndex],
                  key: ValueKey(_tipIndex),
                  textAlign: TextAlign.center,
                  style: const TextStyle(
                    color: OhSheetColors.mutedText,
                    fontSize: 14,
                    fontStyle: FontStyle.italic,
                  ),
                ),
              ),

            // Error state
            if (failed) ...[
              const SizedBox(height: 16),
              Text(
                _failureMessage!,
                textAlign: TextAlign.center,
                style: const TextStyle(color: OhSheetColors.error, fontSize: 14),
              ),
              const SizedBox(height: 16),
              FilledButton(
                onPressed: () => Navigator.of(context).pop(),
                child: const Text('Back'),
              ),
            ],

            const Spacer(),

            // Event log (collapsed, scrollable)
            Expanded(
              child: ListView(
                children: [
                  for (final e in _events.reversed)
                    ListTile(
                      dense: true,
                      leading: _eventIcon(e.type),
                      title: Text(
                        _friendlyEventText(e),
                        style: const TextStyle(fontSize: 13),
                      ),
                      subtitle: e.message == null ? null : Text(
                        e.message!,
                        style: const TextStyle(fontSize: 12, color: OhSheetColors.mutedText),
                      ),
                    ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  String _friendlyEventText(JobEvent e) => switch (e.type) {
        'job_created' => 'Job created',
        'job_started' => 'Pipeline started',
        'stage_started' => '${friendlyStageName(e.stage ?? '')}…',
        'stage_completed' => '${friendlyStageName(e.stage ?? '')} complete',
        'job_succeeded' => 'Complete!',
        'job_failed' => 'Failed',
        _ => e.type,
      };

  Icon _eventIcon(String type) => switch (type) {
        'job_succeeded' || 'stage_completed' => const Icon(Icons.check_circle, color: OhSheetColors.success, size: 18),
        'job_failed' || 'stage_failed' => const Icon(Icons.error, color: OhSheetColors.error, size: 18),
        'stage_started' => const Icon(Icons.play_arrow, color: OhSheetColors.teal, size: 18),
        _ => const Icon(Icons.info_outline, color: OhSheetColors.mutedText, size: 18),
      };
}

// ---------------------------------------------------------------------------
// Stage badge widget
// ---------------------------------------------------------------------------

class _StageBadge extends StatelessWidget {
  const _StageBadge({required this.label, required this.done, required this.active});
  final String label;
  final bool done;
  final bool active;

  @override
  Widget build(BuildContext context) {
    final Color bg;
    final Color fg;
    if (done) {
      bg = OhSheetColors.success;
      fg = Colors.white;
    } else if (active) {
      bg = OhSheetColors.teal;
      fg = Colors.white;
    } else {
      bg = Colors.grey.shade200;
      fg = OhSheetColors.mutedText;
    }

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: bg,
        borderRadius: BorderRadius.circular(16),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (done) ...[
            const Icon(Icons.check, color: Colors.white, size: 14),
            const SizedBox(width: 4),
          ],
          if (active) ...[
            const SizedBox(
              width: 12,
              height: 12,
              child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
            ),
            const SizedBox(width: 4),
          ],
          Text(label, style: TextStyle(color: fg, fontSize: 12, fontWeight: FontWeight.w600)),
        ],
      ),
    );
  }
}
