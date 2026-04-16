/// Result screen — shows resolved song info, sheet music, downloads,
/// and optional TuneChat embedded score viewer.
///
/// Performance: StatefulWidget keyed by jobId, deferred OSMD viewer,
/// RepaintBoundary around the heavy viewer.
library;

import 'dart:math' as math;
import 'dart:ui_web' as ui_web;

import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:web/web.dart' as web;

import '../api/client.dart';
import '../api/models.dart';
import '../responsive.dart';
import '../theme.dart';
import '../widgets/sheet_music_viewer.dart';
import '../widgets/sticker_widgets.dart';

class ResultScreen extends StatefulWidget {
  const ResultScreen({super.key, required this.api, required this.job});
  final OhSheetApi api;
  final JobSummary job;

  @override
  State<ResultScreen> createState() => _ResultScreenState();
}

class _ResultScreenState extends State<ResultScreen> {
  bool _viewerReady = false;

  late final String _musicxmlUrl;
  late final String _midiUrl;
  late final String? _sourceUrl;
  late final String _displayTitle;
  late final String? _displayArtist;
  late final String? _tuneChatJobId;
  late final bool _hasTuneChat;

  static const _tuneChatBaseUrl = String.fromEnvironment(
    'TUNECHAT_URL',
    defaultValue: 'http://localhost:5173',
  );

  @override
  void initState() {
    super.initState();
    final job = widget.job;
    final api = widget.api;

    _musicxmlUrl = api.artifactUrl(job.jobId, 'musicxml');
    _midiUrl = api.artifactUrl(job.jobId, 'midi');

    _displayTitle = (job.title != null && !job.title!.startsWith('http'))
        ? job.title!
        : 'Your piece';
    _displayArtist = job.artist;

    final resultMap = job.result ?? {};
    _sourceUrl = job.sourceUrl;

    _tuneChatJobId = resultMap['tunechat_job_id'] as String?;
    _hasTuneChat = _tuneChatJobId != null;

    SchedulerBinding.instance.addPostFrameCallback((_) {
      if (mounted) setState(() => _viewerReady = true);
    });
  }

  void _download(String kind) {
    final url = widget.api.artifactUrl(widget.job.jobId, kind);
    launchUrl(Uri.parse(url), mode: LaunchMode.externalApplication);
  }

  @override
  Widget build(BuildContext context) {
    final size = MediaQuery.sizeOf(context);
    final sheetH = math.min(700.0, math.max(400.0, size.height * 0.55));
    final twoCol = context.ohSheetResultTwoColumn;

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
          maxWidth: OhSheetBreakpoints.contentWide,
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
          child: SingleChildScrollView(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                _buildHeader(twoCol),
                const SizedBox(height: 24),
                // If TuneChat processed this job, show its embedded
                // score viewer. Otherwise fall back to Oh Sheet's OSMD.
                _hasTuneChat
                    ? _buildTuneChatEmbed(sheetH)
                    : _buildSheetSection(sheetH),
                // TuneChat's iframe ships its own download controls, and
                // the TuneChat short-circuit leaves pdf/midi/musicxml
                // artifact URIs empty — rendering our buttons would 404.
                if (!_hasTuneChat) ...[
                  const SizedBox(height: 20),
                  _buildDownloads(),
                ],
                SizedBox(height: twoCol ? 24 : 32),
                _buildAgainButton(),
              ],
            ),
          ),
        ),
      ),
    );
  }

  // ── Toggle: Oh Sheet | TuneChat ✨ ──────────────────────────────────

  // ── TuneChat embedded iframe ─────────────────────────────────────────
  // When TuneChat processed this job, embed its score viewer as an
  // iframe. TuneChat renders everything — Oh Sheet just provides the URL.
  // The embed route (/embed?job=...) serves a stripped-down viewer with
  // playback, attribution, and an "Open Full Experience" link.

  Widget _buildTuneChatEmbed(double height) {
    // Iframe the full TuneChat app with deep-link params. TuneChat's
    // mountApp() reads ?job= and auto-creates a room with the score
    // loaded — the user lands directly in the listening room.
    final tuneChatUrl = Uri.parse(_tuneChatBaseUrl).replace(
      queryParameters: {
        'job': _tuneChatJobId!,
        'title': _displayTitle,
        if (_displayArtist != null && _displayArtist.isNotEmpty)
          'artist': _displayArtist,
      },
    ).toString();

    final viewType = 'tunechat-app-${widget.job.jobId}';
    ui_web.platformViewRegistry.registerViewFactory(viewType, (int id) {
      final iframe = web.document.createElement('iframe') as web.HTMLIFrameElement;
      iframe.src = tuneChatUrl;
      iframe.style.cssText = 'width:100%;height:100%;border:none;border-radius:12px;';
      iframe.allow = 'autoplay';
      return iframe;
    });

    return SizedBox(
      height: math.max(600.0, height),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(12),
        child: HtmlElementView(viewType: viewType),
      ),
    );
  }

  // ── Header: mascot + title + artist + YouTube link ──────────────────

  Widget _buildHeader(bool twoCol) {
    return Column(
      children: [
        SvgPicture.asset(
          'assets/mascots/mascot-success.svg',
          height: twoCol ? 120 : 160,
          fit: BoxFit.contain,
        ),
        const SizedBox(height: 16),
        Text(
          _displayTitle,
          textAlign: TextAlign.center,
          style: const TextStyle(
            fontSize: 22,
            fontWeight: FontWeight.w700,
            color: OhSheetColors.darkText,
          ),
        ),
        if (_displayArtist != null && _displayArtist.isNotEmpty) ...[
          const SizedBox(height: 4),
          Text(
            _displayArtist,
            textAlign: TextAlign.center,
            style: const TextStyle(
              fontSize: 16,
              color: OhSheetColors.mutedText,
            ),
          ),
        ],
        if (_sourceUrl case final url?) ...[
          const SizedBox(height: 8),
          GestureDetector(
            onTap: () => launchUrl(
              Uri.parse(url),
              mode: LaunchMode.externalApplication,
            ),
            child: const Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(Icons.play_circle_outline, size: 16, color: OhSheetColors.teal),
                SizedBox(width: 4),
                Text(
                  'View on YouTube',
                  style: TextStyle(
                    fontSize: 13,
                    color: OhSheetColors.teal,
                    decoration: TextDecoration.underline,
                    decorationColor: OhSheetColors.teal,
                  ),
                ),
              ],
            ),
          ),
        ],
      ],
    );
  }

  // ── Oh Sheet score viewer (deferred for perf) ───────────────────────

  Widget _buildSheetSection(double height) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const OhSheetStickerSectionTitle(text: 'Sheet Music'),
        const SizedBox(height: 10),
        RepaintBoundary(
          child: OhSheetStickerClip(
            height: height,
            child: _viewerReady
                ? SheetMusicViewer(
                    key: ValueKey('sheet-${widget.job.jobId}'),
                    musicxmlUrl: _musicxmlUrl,
                    midiUrl: _midiUrl,
                  )
                : const Center(
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        CircularProgressIndicator(
                          strokeWidth: 2,
                          color: OhSheetColors.teal,
                        ),
                        SizedBox(height: 12),
                        Text(
                          'Loading score…',
                          style: TextStyle(
                            color: OhSheetColors.mutedText,
                            fontSize: 13,
                          ),
                        ),
                      ],
                    ),
                  ),
          ),
        ),
      ],
    );
  }

  // ── Downloads ───────────────────────────────────────────────────────

  Widget _buildDownloads() {
    return LayoutBuilder(
      builder: (context, constraints) {
        final buttons = [
          _dlButton('PDF', Icons.picture_as_pdf, 'pdf', filled: true),
          _dlButton('MIDI', Icons.music_note, 'midi', filled: true),
          _dlButton('MusicXML', Icons.notes, 'musicxml', filled: false),
        ];
        if (constraints.maxWidth < 340) {
          return Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              for (var i = 0; i < buttons.length; i++) ...[
                if (i > 0) const SizedBox(height: 8),
                buttons[i],
              ],
            ],
          );
        }
        return Row(
          children: [
            for (var i = 0; i < buttons.length; i++) ...[
              if (i > 0) const SizedBox(width: 8),
              Expanded(child: buttons[i]),
            ],
          ],
        );
      },
    );
  }

  Widget _dlButton(String label, IconData icon, String kind, {required bool filled}) {
    if (filled) {
      return FilledButton.icon(
        onPressed: () => _download(kind),
        icon: Icon(icon, size: 18),
        label: Text(label),
      );
    }
    return OutlinedButton.icon(
      onPressed: () => _download(kind),
      icon: Icon(icon, size: 18),
      label: Text(label),
    );
  }

  // ── Again button ────────────────────────────────────────────────────

  Widget _buildAgainButton() {
    return SizedBox(
      width: double.infinity,
      child: OutlinedButton.icon(
        onPressed: () => Navigator.of(context).popUntil((r) => r.isFirst),
        icon: const Icon(Icons.refresh),
        label: const Text('Transcribe another song'),
      ),
    );
  }
}
