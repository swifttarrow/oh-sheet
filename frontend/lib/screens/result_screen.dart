/// Result screen — shows the success mascot, song info, and download buttons.
library;

import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:url_launcher/url_launcher.dart';

import '../api/client.dart';
import '../api/models.dart';
import '../responsive.dart';
import '../theme.dart';
import '../widgets/midi_player.dart';
import '../widgets/sheet_music_viewer.dart';
import '../widgets/sticker_widgets.dart';

class ResultScreen extends StatelessWidget {
  const ResultScreen({super.key, required this.api, required this.job});
  final OhSheetApi api;
  final JobSummary job;

  void _download(String kind) {
    final url = api.artifactUrl(job.jobId, kind);
    launchUrl(Uri.parse(url), mode: LaunchMode.externalApplication);
  }

  @override
  Widget build(BuildContext context) {
    final size = MediaQuery.sizeOf(context);
    final sheetH = math.min(700.0, math.max(400.0, size.height * 0.7));
    final twoCol = context.ohSheetResultTwoColumn;

    final header = Column(
      children: [
        SvgPicture.asset(
          'assets/mascots/mascot-success.svg',
          height: twoCol ? 120 : 160,
          fit: BoxFit.contain,
        ),
        const SizedBox(height: 16),
        if (job.title != null)
          Text(
            job.title!,
            textAlign: TextAlign.center,
            style: const TextStyle(
              fontSize: 22,
              fontWeight: FontWeight.w700,
              color: OhSheetColors.darkText,
            ),
          ),
        if (job.artist != null) ...[
          const SizedBox(height: 4),
          Text(
            job.artist!,
            textAlign: TextAlign.center,
            style: const TextStyle(
              fontSize: 16,
              color: OhSheetColors.mutedText,
            ),
          ),
        ],
      ],
    );

    final sheetSection = Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const OhSheetStickerSectionTitle(text: 'Sheet Music'),
        const SizedBox(height: 10),
        OhSheetStickerClip(
          height: sheetH,
          child: SheetMusicViewer(
            musicxmlUrl: api.artifactUrl(job.jobId, 'musicxml'),
            midiUrl: api.artifactUrl(job.jobId, 'midi'),
          ),
        ),
      ],
    );

    final listenSection = Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const OhSheetStickerSectionTitle(
          text: 'Listen',
          accent: OhSheetColors.orange,
        ),
        const SizedBox(height: 10),
        OhSheetStickerClip(
          height: math.min(280.0, math.max(160.0, size.height * 0.25)),
          child: MidiPlayerWidget(midiUrl: api.artifactUrl(job.jobId, 'midi')),
        ),
      ],
    );

    final downloads = LayoutBuilder(
      builder: (context, constraints) {
        if (constraints.maxWidth < 340) {
          return Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              FilledButton.icon(
                onPressed: () => _download('pdf'),
                icon: const Icon(Icons.picture_as_pdf, size: 18),
                label: const Text('PDF'),
              ),
              const SizedBox(height: 8),
              FilledButton.icon(
                onPressed: () => _download('midi'),
                icon: const Icon(Icons.music_note, size: 18),
                label: const Text('MIDI'),
              ),
              const SizedBox(height: 8),
              OutlinedButton.icon(
                onPressed: () => _download('musicxml'),
                icon: const Icon(Icons.notes, size: 18),
                label: const Text('MusicXML'),
              ),
            ],
          );
        }
        return Row(
          children: [
            Expanded(
              child: FilledButton.icon(
                onPressed: () => _download('pdf'),
                icon: const Icon(Icons.picture_as_pdf, size: 18),
                label: const Text('PDF'),
              ),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: FilledButton.icon(
                onPressed: () => _download('midi'),
                icon: const Icon(Icons.music_note, size: 18),
                label: const Text('MIDI'),
              ),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: OutlinedButton.icon(
                onPressed: () => _download('musicxml'),
                icon: const Icon(Icons.notes, size: 18),
                label: const Text('MusicXML'),
              ),
            ),
          ],
        );
      },
    );

    final againButton = SizedBox(
      width: double.infinity,
      child: OutlinedButton.icon(
        onPressed: () => Navigator.of(context).popUntil((r) => r.isFirst),
        icon: const Icon(Icons.refresh),
        label: const Text('Transcribe another song'),
      ),
    );

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
            child: twoCol
                ? Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      header,
                      const SizedBox(height: 24),
                      sheetSection,
                      const SizedBox(height: 24),
                      listenSection,
                      const SizedBox(height: 20),
                      downloads,
                      const SizedBox(height: 24),
                      againButton,
                    ],
                  )
                : Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      header,
                      const SizedBox(height: 24),
                      sheetSection,
                      const SizedBox(height: 24),
                      listenSection,
                      const SizedBox(height: 24),
                      downloads,
                      const SizedBox(height: 32),
                      againButton,
                    ],
                  ),
          ),
        ),
      ),
    );
  }
}
