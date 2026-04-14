/// Live progress screen — subscribes to /v1/jobs/{id}/ws and renders the
/// pipeline with stage-specific mascot images, sticker badges, and rotating tips.
library;

import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';

import '../api/client.dart';
import '../api/models.dart';
import '../api/ws.dart';
import '../responsive.dart';
import '../theme.dart';
import '../widgets/sticker_widgets.dart';
import 'result_screen.dart';

// ---------------------------------------------------------------------------
// Public helpers (tested independently)
// ---------------------------------------------------------------------------

String friendlyStageName(String stage) => switch (stage) {
      'ingest' => 'Preparing',
      'transcribe' => 'Transcribing',
      'arrange' => 'Arranging',
      'humanize' => 'Humanizing',
      'refine' => 'Refining',
      'engrave' => 'Engraving',
      _ => stage,
    };

String mascotAssetForStage(String? stage) => switch (stage) {
      'ingest' => 'assets/mascots/mascot-progress-ingest.svg',
      'transcribe' => 'assets/mascots/mascot-progress-transcribe.svg',
      'arrange' || 'humanize' || 'refine' => 'assets/mascots/mascot-progress-arrange.svg',
      'engrave' => 'assets/mascots/mascot-progress-engrave.svg',
      _ => 'assets/mascots/mascot-progress-ingest.svg',
    };

const pipelineTips = [
  'Most songs take 15–45 seconds.',
  'The AI is analyzing rhythm, melody, and harmony.',
  'Your piano arrangement will have right and left hand parts.',
  'Difficulty is rated automatically from 1–10.',
  'The final PDF is typeset with LilyPond — publication quality.',
];

/// Returns the ordered stage list the progress screen should DISPLAY and
/// use as the denominator when computing the progress fraction.
///
/// When [enableRefine] is true, the 'refine' stage slots between 'arrange'
/// and 'engrave' (matching backend PipelineConfig.get_execution_plan()).
/// When false, returns the base 5-stage list — identical to the pre-Phase-3
/// shape so unrefined jobs still reach 1.0 progress at completion.
///
/// 'humanize' is deliberately hidden from the badge row (its mascot reuses
/// arrange's), but it IS still counted by [computeProgress] — the
/// denominator there needs all pipeline stages, not just the visible ones.
///
/// Why not mutate the static `kPipelineStages` in models.dart? Because
/// kPipelineStages.length is used as a progress fallback denominator and
/// the DEFAULT post-Phase-3 path is `enable_refine=false`. Bumping the
/// static list to 6 would cap unrefined-job progress at 5/6 ≈ 0.83 — a
/// user-visible regression on the dominant flow for an opt-in feature.
/// The denominator must adapt to the job's actual execution plan.
List<String> expectedStagesFor({required bool enableRefine}) {
  if (enableRefine) {
    return const ['ingest', 'transcribe', 'arrange', 'refine', 'engrave'];
  }
  return const ['ingest', 'transcribe', 'arrange', 'engrave'];
}

/// Full ordered pipeline stage list (for progress-fraction denominator) —
/// this is the complete set of stages the runner will execute for the
/// given [enableRefine] flag, including 'humanize' which is hidden from
/// the badge row but still emits stage_completed events.
List<String> _pipelineStagesFor({required bool enableRefine}) {
  if (enableRefine) {
    return const ['ingest', 'transcribe', 'arrange', 'humanize', 'refine', 'engrave'];
  }
  return const ['ingest', 'transcribe', 'arrange', 'humanize', 'engrave'];
}

/// Pure helper: fraction of expected pipeline stages that are in [completed].
///
/// Exposed publicly so widget tests can exercise the default-path invariant
/// (completion of all base stages → 1.0) without needing JobEventStream
/// substitution. For an UNREFINED job (the default, dominant path),
/// completion of all 5 base stages yields 1.0 — NOT 5/6 ≈ 0.83 which
/// would happen if the denominator were forced to 6.
double computeProgress({
  required Iterable<String> completed,
  required bool enableRefine,
}) {
  final expected = _pipelineStagesFor(enableRefine: enableRefine);
  if (expected.isEmpty) return 0.0;
  final expectedSet = expected.toSet();
  final completedInExpected =
      completed.where(expectedSet.contains).toSet().length;
  return (completedInExpected / expected.length).clamp(0.0, 1.0);
}

// ---------------------------------------------------------------------------
// Screen
// ---------------------------------------------------------------------------

class ProgressScreen extends StatefulWidget {
  const ProgressScreen({
    super.key,
    required this.api,
    required this.jobId,
    this.enableRefine = false,
  });
  final OhSheetApi api;
  final String jobId;

  /// Whether the job was submitted with `enable_refine=true`. Drives the
  /// badge row (shows the 'refine' badge between arrange and engrave)
  /// and the progress-fraction denominator (6 vs 5 stages). Defaults to
  /// false so existing callers that don't yet thread the flag through
  /// (pre-Plan-04 upload screen) keep the default 5-stage behavior.
  final bool enableRefine;

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
  // UX-04: set when the refine stage emits `stage_completed` with a
  // `refine_skipped: <reason>` message. Drives the badge-row swap to
  // RefineSkippedBadge (amber) instead of the regular _StageBadge.
  bool _refineSkipped = false;

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
          // UX-04: detect the refine skip path. The runner emits
          // `message="refine_skipped: <reason>"` on stage_completed when
          // the LLM call failed and engrave will run on the unrefined
          // payload. Flag it so the badge row swaps in the amber
          // RefineSkippedBadge instead of the green done state.
          if (event.stage == 'refine' &&
              (event.message?.startsWith('refine_skipped:') ?? false)) {
            _refineSkipped = true;
          }
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
    // I-05: the denominator adapts to the job's actual execution plan.
    // Unrefined (default) jobs see a 5-stage denominator; refined jobs
    // see 6. Mutating the static `kPipelineStages` would cap the default
    // path at 5/6 ≈ 0.83 — a user-visible regression on the dominant
    // post-Phase-3 flow for an opt-in feature.
    return computeProgress(
      completed: _completedStages,
      enableRefine: widget.enableRefine,
    );
  }

  @override
  Widget build(BuildContext context) {
    final failed = _failureMessage != null;
    final orderedEvents = _events.reversed.toList(growable: false);

    return Scaffold(
      backgroundColor: OhSheetColors.cream,
      appBar: AppBar(
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () => Navigator.of(context).pop(),
        ),
        title: const Text('Oh Sheet!'),
      ),
      body: SafeArea(
        child: OhSheetResponsiveBody(
          maxWidth: OhSheetBreakpoints.contentMedium,
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
          child: CustomScrollView(
            slivers: [
              SliverToBoxAdapter(
                child: OhSheetSticker(
                  child: Column(
                    children: [
                      AnimatedSwitcher(
                        duration: const Duration(milliseconds: 400),
                        child: SvgPicture.asset(
                          failed
                              ? 'assets/mascots/mascot-error.svg'
                              : mascotAssetForStage(_currentStage),
                          key: ValueKey(failed ? 'error' : _currentStage),
                          height: 176,
                          fit: BoxFit.contain,
                        ),
                      ),
                      const SizedBox(height: 18),
                      Wrap(
                        alignment: WrapAlignment.center,
                        spacing: 8,
                        runSpacing: 10,
                        children: [
                          for (final stage in expectedStagesFor(
                              enableRefine: widget.enableRefine))
                            if (stage == 'refine' && _refineSkipped)
                              const RefineSkippedBadge()
                            else
                              _StageBadge(
                                label: friendlyStageName(stage),
                                done: _completedStages.contains(stage),
                                active: _currentStage == stage &&
                                    !_completedStages.contains(stage),
                              ),
                        ],
                      ),
                      const SizedBox(height: 22),
                      Container(
                        height: 20,
                        decoration: BoxDecoration(
                          borderRadius: BorderRadius.circular(14),
                          border: Border.all(color: OhSheetColors.inkStroke, width: 2.5),
                          color: Colors.grey.shade200,
                        ),
                        clipBehavior: Clip.antiAlias,
                        child: LinearProgressIndicator(
                          value: failed ? null : _progress,
                          minHeight: 20,
                          backgroundColor: Colors.transparent,
                          valueColor: AlwaysStoppedAnimation(
                            failed ? OhSheetColors.error : OhSheetColors.teal,
                          ),
                        ),
                      ),
                      const SizedBox(height: 20),
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
                              fontWeight: FontWeight.w600,
                              fontStyle: FontStyle.italic,
                            ),
                          ),
                        ),
                      if (failed) ...[
                        const SizedBox(height: 8),
                        Text(
                          _failureMessage!,
                          textAlign: TextAlign.center,
                          style: const TextStyle(
                            color: OhSheetColors.error,
                            fontSize: 14,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                        const SizedBox(height: 14),
                        FilledButton(
                          onPressed: () => Navigator.of(context).pop(),
                          child: const Text('Back'),
                        ),
                      ],
                    ],
                  ),
                ),
              ),
              if (orderedEvents.isNotEmpty) ...[
                const SliverToBoxAdapter(child: SizedBox(height: 20)),
                const SliverToBoxAdapter(
                  child: OhSheetStickerSectionTitle(
                    text: 'Activity',
                    accent: OhSheetColors.orange,
                  ),
                ),
                const SliverToBoxAdapter(child: SizedBox(height: 10)),
              ],
              if (orderedEvents.isNotEmpty)
                SliverList(
                  delegate: SliverChildBuilderDelegate(
                    (context, index) {
                      final e = orderedEvents[index];
                      return ListTile(
                        dense: true,
                        leading: _eventIcon(e.type),
                        title: Text(
                          _friendlyEventText(e),
                          style: const TextStyle(fontSize: 13),
                        ),
                        subtitle: e.message == null
                            ? null
                            : Text(
                                e.message!,
                                style: const TextStyle(
                                  fontSize: 12,
                                  color: OhSheetColors.mutedText,
                                ),
                              ),
                      );
                    },
                    childCount: orderedEvents.length,
                  ),
                ),
              const SliverToBoxAdapter(child: SizedBox(height: 24)),
            ],
          ),
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
      bg = Colors.white;
      fg = OhSheetColors.mutedText;
    }

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: bg,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: OhSheetColors.inkStroke, width: 2),
        boxShadow: [
          BoxShadow(
            color: OhSheetColors.inkStroke.withValues(alpha: 0.06),
            offset: const Offset(2, 3),
            blurRadius: 0,
          ),
        ],
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
          Text(label, style: TextStyle(color: fg, fontSize: 12, fontWeight: FontWeight.w800)),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Refine-skipped badge (UX-04)
// ---------------------------------------------------------------------------

/// UX-04: visually distinct badge shown when the refine stage emits a
/// `stage_completed` event with `message="refine_skipped: <reason>"`.
///
/// NOT red (that would imply failure — the job itself succeeded and the
/// user is still getting a PDF). Uses the amber/orange brand accent
/// (`OhSheetColors.orange`) to signal "feature degraded" without
/// implying the pipeline is broken.
///
/// Public so widget tests can pump it directly (see
/// `frontend/test/widgets/progress_screen_refine_test.dart`) — avoids
/// needing to substitute JobEventStream in tests.
class RefineSkippedBadge extends StatelessWidget {
  const RefineSkippedBadge({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: OhSheetColors.orange,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: OhSheetColors.inkStroke, width: 2),
        boxShadow: [
          BoxShadow(
            color: OhSheetColors.inkStroke.withValues(alpha: 0.06),
            offset: const Offset(2, 3),
            blurRadius: 0,
          ),
        ],
      ),
      child: const Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.info_outline, color: Colors.white, size: 14),
          SizedBox(width: 4),
          Text(
            'Refinement unavailable',
            style: TextStyle(
              color: Colors.white,
              fontSize: 12,
              fontWeight: FontWeight.w800,
            ),
          ),
        ],
      ),
    );
  }
}
