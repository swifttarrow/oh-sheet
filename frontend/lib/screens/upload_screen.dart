/// Upload screen — pick audio, MIDI, or type a song title, then submit a job.
library;

import 'dart:typed_data';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';

import '../api/client.dart';
import '../api/models.dart';
import '../responsive.dart';
import '../theme.dart';
import '../widgets/sticker_widgets.dart';
import 'progress_screen.dart';

enum _SourceMode { audio, midi, title, youtube }

// Phase 3 (WR-02): distinguish "probe in flight" from "probe said no" from
// "probe never completed (network failure)". Conflating the last two into a
// single `refineAvailable=false` signal hides airplane-mode / backend-down
// failures behind the "not configured on this server" helper text, leaving
// the user with no retry affordance and a misleading diagnosis.
enum _CapabilitiesState { loading, available, notConfigured, probeFailed }

class UploadScreen extends StatefulWidget {
  const UploadScreen({super.key, required this.api});
  final OhSheetApi api;

  @override
  State<UploadScreen> createState() => _UploadScreenState();
}

class _UploadScreenState extends State<UploadScreen> {
  _SourceMode _mode = _SourceMode.audio;
  final _titleController = TextEditingController();
  final _artistController = TextEditingController();
  final _youtubeController = TextEditingController();

  PlatformFile? _pickedFile;
  bool _submitting = false;
  String? _error;

  // Opt-in for the backend cover_search fast path. Only surfaced in
  // YouTube mode because it's meaningless for audio uploads (user
  // already picked the source) and plain title lookups (no URL to swap).
  // Defaults off so the user's first experience matches what they
  // pasted; they can flip it on once they see the option.
  bool _preferCleanSource = false;

  // Phase 3 (D-23): refine opt-in state. Defaults false on every screen
  // construction and is NEVER persisted across sessions — the user must
  // re-opt-in every upload. Matches PROJECT.md's default-off-until-quality-
  // proven posture.
  bool _enableRefine = false;

  // Phase 3 (D-22, WR-02): capabilities probe — populated once at
  // initState(). While loading we optimistically treat refine as available
  // so a slow network doesn't flicker the toggle into the disabled state.
  // WR-02: we track three terminal states (available, notConfigured,
  // probeFailed) so "server said no" and "couldn't reach server" render
  // different helper text and have different enablement — probeFailed
  // keeps the toggle enabled so a real submit-time error can surface
  // instead of the misleading "not configured" message.
  _CapabilitiesState _capabilitiesState = _CapabilitiesState.loading;

  static final _youtubeRegex = RegExp(
    r'^https?://(www\.|music\.|m\.)?youtu(\.be/|be\.com/watch\?v=)([\w-]{11})',
  );

  bool get _isValidYoutubeUrl => _youtubeRegex.hasMatch(_youtubeController.text.trim());

  String? get _youtubeValidationError {
    final text = _youtubeController.text.trim();
    if (text.isEmpty) return null;
    if (!_isValidYoutubeUrl) return 'Enter a valid YouTube URL';
    return null;
  }

  @override
  void initState() {
    super.initState();
    _loadCapabilities();
  }

  Future<void> _loadCapabilities() async {
    try {
      final caps = await widget.api.getCapabilities();
      if (!mounted) return;
      setState(() => _capabilitiesState = caps.refineAvailable
          ? _CapabilitiesState.available
          : _CapabilitiesState.notConfigured);
    } catch (_) {
      // WR-02: probe failure (network unreachable, 500, malformed body) is
      // NOT the same signal as "server responded refineAvailable=false".
      // We surface a distinct state so the UI can render different helper
      // text and keep the toggle enabled, letting a submission attempt
      // surface the real backend error instead of a misleading
      // "not configured" diagnosis.
      if (!mounted) return;
      setState(() => _capabilitiesState = _CapabilitiesState.probeFailed);
    }
  }

  @override
  void dispose() {
    _titleController.dispose();
    _artistController.dispose();
    _youtubeController.dispose();
    super.dispose();
  }

  Future<void> _pick() async {
    final isAudio = _mode == _SourceMode.audio;
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: isAudio
          ? const ['mp3', 'wav', 'flac', 'm4a']
          : const ['mid', 'midi'],
      withData: true,
    );
    if (result == null || result.files.isEmpty) return;
    setState(() {
      _pickedFile = result.files.first;
      _error = null;
    });
  }

  Future<void> _submit() async {
    setState(() {
      _submitting = true;
      _error = null;
    });

    try {
      JobSummary job;
      switch (_mode) {
        case _SourceMode.audio:
          if (_pickedFile == null) throw StateError('Pick an audio file first');
          final bytes = _pickedFile!.bytes;
          if (bytes == null) throw StateError('File bytes unavailable on this platform');
          final audio = await widget.api.uploadAudio(
            bytes: Uint8List.fromList(bytes),
            filename: _pickedFile!.name,
          );
          job = await widget.api.createJob(
            audio: audio,
            title: _titleController.text.trim().isEmpty
                ? null
                : _titleController.text.trim(),
            artist: _artistController.text.trim().isEmpty
                ? null
                : _artistController.text.trim(),
            enableRefine: _enableRefine,
          );
          break;
        case _SourceMode.midi:
          if (_pickedFile == null) throw StateError('Pick a MIDI file first');
          final bytes = _pickedFile!.bytes;
          if (bytes == null) throw StateError('File bytes unavailable on this platform');
          final midi = await widget.api.uploadMidi(
            bytes: Uint8List.fromList(bytes),
            filename: _pickedFile!.name,
          );
          job = await widget.api.createJob(
            midi: midi,
            title: _titleController.text.trim().isEmpty
                ? null
                : _titleController.text.trim(),
            artist: _artistController.text.trim().isEmpty
                ? null
                : _artistController.text.trim(),
            enableRefine: _enableRefine,
          );
          break;
        case _SourceMode.title:
          final title = _titleController.text.trim();
          if (title.isEmpty) throw StateError('Enter a song title');
          job = await widget.api.createJob(
            title: title,
            artist: _artistController.text.trim().isEmpty
                ? null
                : _artistController.text.trim(),
            enableRefine: _enableRefine,
          );
          break;
        case _SourceMode.youtube:
          final url = _youtubeController.text.trim();
          if (url.isEmpty) throw StateError('Enter a YouTube URL');
          if (!_isValidYoutubeUrl) throw StateError('Enter a valid YouTube URL');
          job = await widget.api.createJob(
            title: url,
            artist: _artistController.text.trim().isEmpty
                ? null
                : _artistController.text.trim(),
            preferCleanSource: _preferCleanSource,
            enableRefine: _enableRefine,
          );
          break;
      }

      if (!mounted) return;
      await Navigator.of(context).push(
        MaterialPageRoute(
          builder: (_) => ProgressScreen(
            api: widget.api,
            jobId: job.jobId,
            enableRefine: _enableRefine,
          ),
        ),
      );
    } catch (e) {
      setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _submitting = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final needsFile = _mode == _SourceMode.audio || _mode == _SourceMode.midi;
    final canSubmit = !_submitting &&
        switch (_mode) {
          _SourceMode.title => _titleController.text.trim().isNotEmpty,
          _SourceMode.youtube => _isValidYoutubeUrl,
          _ => _pickedFile != null,
        };

    return Scaffold(
      backgroundColor: OhSheetColors.cream,
      body: SafeArea(
        child: OhSheetResponsiveBody(
          maxWidth: OhSheetBreakpoints.contentMedium,
          padding: const EdgeInsets.fromLTRB(16, 20, 16, 24),
          child: SingleChildScrollView(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Center(
                  child: SvgPicture.asset(
                    'assets/mascots/mascot-home-happy.svg',
                    height: 132,
                    fit: BoxFit.contain,
                    clipBehavior: Clip.none,
                    allowDrawingOutsideViewBox: true,
                  ),
                ),
                const SizedBox(height: 14),
                Text(
                  'Turn any song into piano sheet music',
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    fontSize: 21,
                    fontWeight: FontWeight.w800,
                    color: OhSheetColors.darkText,
                    height: 1.25,
                    shadows: [
                      Shadow(
                        color: OhSheetColors.orange.withValues(alpha: 0.35),
                        offset: const Offset(0, 2),
                        blurRadius: 0,
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 6),
                const Text(
                  'Upload audio, paste a YouTube link, or drop a MIDI file.',
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    fontSize: 14,
                    color: OhSheetColors.mutedText,
                    fontWeight: FontWeight.w500,
                  ),
                ),
                const SizedBox(height: 8),
                const Text(
                  'Let’s get sheet music! 🎹',
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    fontSize: 13,
                    fontWeight: FontWeight.w600,
                    color: OhSheetColors.pinkAccent,
                  ),
                ),
                const SizedBox(height: 22),
                OhSheetSticker(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      SingleChildScrollView(
                        scrollDirection: Axis.horizontal,
                        child: SegmentedButton<_SourceMode>(
                          segments: const [
                            ButtonSegment(value: _SourceMode.audio, label: Text('Audio')),
                            ButtonSegment(value: _SourceMode.midi, label: Text('MIDI')),
                            ButtonSegment(value: _SourceMode.title, label: Text('Title')),
                            ButtonSegment(value: _SourceMode.youtube, label: Text('YouTube')),
                          ],
                          selected: {_mode},
                          onSelectionChanged: (s) => setState(() {
                            _mode = s.first;
                            _pickedFile = null;
                            _error = null;
                            // Reset clean-source opt-in on every mode
                            // change. The toggle is YouTube-only; letting
                            // its state persist across a mode switch
                            // means the user could flip it ON, switch
                            // to Audio, switch back to YouTube, and
                            // submit with the flag still set silently —
                            // the invisible-state footgun PR #47 review
                            // flagged as (Important).
                            _preferCleanSource = false;
                          }),
                        ),
                      ),
                      const SizedBox(height: 22),
                      if (needsFile) ...[
                        OhSheetDashedPickZone(
                          onTap: _submitting ? null : _pick,
                          child: Padding(
                            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 20),
                            child: Column(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                Icon(
                                  Icons.add_circle_outline,
                                  size: 40,
                                  color: OhSheetColors.teal.withValues(alpha: 0.9),
                                ),
                                const SizedBox(height: 10),
                                Text(
                                  _pickedFile == null
                                      ? (_mode == _SourceMode.audio
                                          ? 'Tap to pick audio'
                                          : 'Tap to pick MIDI')
                                      : _pickedFile!.name,
                                  textAlign: TextAlign.center,
                                  style: const TextStyle(
                                    fontWeight: FontWeight.w800,
                                    fontSize: 15,
                                    color: OhSheetColors.darkText,
                                  ),
                                ),
                                const SizedBox(height: 4),
                                Text(
                                  _mode == _SourceMode.audio
                                      ? 'mp3 · wav · flac · m4a'
                                      : '.mid · .midi',
                                  style: const TextStyle(
                                    fontSize: 12,
                                    color: OhSheetColors.mutedText,
                                    fontWeight: FontWeight.w600,
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ),
                        const SizedBox(height: 18),
                      ],
                      if (_mode == _SourceMode.youtube) ...[
                        TextField(
                          controller: _youtubeController,
                          decoration: InputDecoration(
                            labelText: 'YouTube URL',
                            hintText: 'https://youtube.com/watch?v=...',
                            errorText: _youtubeValidationError,
                            prefixIcon: const Icon(Icons.play_circle_outline),
                          ),
                          onChanged: (_) => setState(() {}),
                        ),
                        const SizedBox(height: 12),
                        TextField(
                          controller: _artistController,
                          decoration: const InputDecoration(
                            labelText: 'Artist (optional)',
                          ),
                        ),
                        const SizedBox(height: 12),
                        // Clean-source opt-in. When on, the backend searches
                        // for the best alternative source (easy/moderate piano
                        // cover OR 8-bit chiptune cover) and transcribes that
                        // instead of the full-band original. Dramatically
                        // cleaner output on pop mixes because Basic Pitch is
                        // much happier with piano-shaped or chiptune-shaped
                        // audio than with a full band.
                        SwitchListTile(
                          key: const ValueKey('ohsheet_prefer_clean_source_toggle'),
                          contentPadding: EdgeInsets.zero,
                          dense: true,
                          value: _preferCleanSource,
                          onChanged: (v) => setState(() => _preferCleanSource = v),
                          activeThumbColor: OhSheetColors.teal,
                          title: const Text(
                            'Find a clean source',
                            style: TextStyle(
                              fontWeight: FontWeight.w700,
                              fontSize: 14,
                              color: OhSheetColors.darkText,
                            ),
                          ),
                          subtitle: const Text(
                            'Search for a piano cover or 8-bit version of '
                            'this song and transcribe that instead — much '
                            'cleaner results for full-band pop tracks.',
                            style: TextStyle(
                              fontSize: 12,
                              color: OhSheetColors.mutedText,
                            ),
                          ),
                        ),
                      ] else ...[
                        TextField(
                          controller: _titleController,
                          decoration: InputDecoration(
                            labelText: _mode == _SourceMode.title
                                ? 'Song title (required)'
                                : 'Title (optional)',
                          ),
                          onChanged: (_) => setState(() {}),
                        ),
                        const SizedBox(height: 12),
                        TextField(
                          controller: _artistController,
                          decoration: const InputDecoration(
                            labelText: 'Artist (optional)',
                          ),
                        ),
                      ],
                      const SizedBox(height: 18),
                      // Phase 3 (D-20): dedicated "AI refinement" section,
                      // visible for all source variants. Distinct from the
                      // _preferCleanSource cluster above because refine is
                      // variant-independent. D-22: disabled with helper text
                      // when the server has no Anthropic key configured.
                      const OhSheetStickerSectionTitle(
                        text: 'AI refinement',
                        accent: OhSheetColors.teal,
                      ),
                      const SizedBox(height: 8),
                      SwitchListTile(
                        key: const ValueKey('enableRefineToggle'),
                        contentPadding: EdgeInsets.zero,
                        dense: true,
                        value: _enableRefine,
                        // WR-02: only disable when the server explicitly
                        // said "not configured". On probeFailed we leave
                        // the toggle enabled so a submit-time error can
                        // surface the real backend issue.
                        onChanged:
                            (_capabilitiesState == _CapabilitiesState.notConfigured)
                                ? null
                                : (v) => setState(() => _enableRefine = v),
                        activeThumbColor: OhSheetColors.teal,
                        title: const Text(
                          'Use AI refinement (experimental)',
                          style: TextStyle(
                            fontWeight: FontWeight.w700,
                            fontSize: 14,
                            color: OhSheetColors.darkText,
                          ),
                        ),
                        subtitle: const Text(
                          'Uses an AI model to polish the generated score. '
                          'Experimental — may add processing cost and a few '
                          'seconds of latency.',
                          style: TextStyle(
                            fontSize: 12,
                            color: OhSheetColors.mutedText,
                          ),
                        ),
                      ),
                      if (_capabilitiesState == _CapabilitiesState.notConfigured)
                        const Padding(
                          padding: EdgeInsets.only(top: 4, left: 4),
                          child: Text(
                            'AI refinement not configured on this server',
                            style: TextStyle(
                              fontSize: 12,
                              color: OhSheetColors.mutedText,
                              fontStyle: FontStyle.italic,
                            ),
                          ),
                        ),
                      if (_capabilitiesState == _CapabilitiesState.probeFailed)
                        const Padding(
                          padding: EdgeInsets.only(top: 4, left: 4),
                          child: Text(
                            'Could not reach the server to check AI '
                            'refinement availability — try again in a moment.',
                            style: TextStyle(
                              fontSize: 12,
                              color: OhSheetColors.mutedText,
                              fontStyle: FontStyle.italic,
                            ),
                          ),
                        ),
                      const SizedBox(height: 22),
                      OhSheetStickerCTA(
                        key: const ValueKey('ohsheet_primary_submit'),
                        onPressed: canSubmit ? _submit : null,
                        loading: _submitting,
                        icon: Icons.play_arrow_rounded,
                        label: _submitting ? 'Working on it…' : "Let's go!",
                      ),
                      if (_error != null) ...[
                        const SizedBox(height: 16),
                        Text(
                          _error!,
                          textAlign: TextAlign.center,
                          style: const TextStyle(
                            color: OhSheetColors.error,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ],
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
