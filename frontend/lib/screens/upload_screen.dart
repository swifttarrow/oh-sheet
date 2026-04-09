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
          );
          break;
      }

      if (!mounted) return;
      await Navigator.of(context).push(
        MaterialPageRoute(
          builder: (_) => ProgressScreen(api: widget.api, jobId: job.jobId),
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
