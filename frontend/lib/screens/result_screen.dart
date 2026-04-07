/// Result screen — shows the success mascot, song info, and download buttons.
library;

import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';

import '../api/client.dart';
import '../api/models.dart';
import '../theme.dart';
import '../widgets/midi_player.dart';
import '../widgets/pdf_preview.dart';

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
    return Scaffold(
      appBar: AppBar(
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () => Navigator.of(context).pop(),
        ),
        title: const Text('Oh Sheet'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
        child: Column(
          children: [
            // Success mascot
            Image.asset(
              'assets/mascots/mascot-success.png',
              height: 160,
            ),
            const SizedBox(height: 16),

            // Song info
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
            const SizedBox(height: 24),

            // Sheet music preview section
            const Align(
              alignment: Alignment.centerLeft,
              child: Text(
                'Sheet Music',
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.w700,
                  color: OhSheetColors.darkText,
                ),
              ),
            ),
            const SizedBox(height: 8),
            Container(
              height: 500,
              width: double.infinity,
              decoration: BoxDecoration(
                color: Colors.white,
                border: Border.all(color: Colors.grey.shade300),
                borderRadius: BorderRadius.circular(12),
              ),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(12),
                child: PdfPreviewWidget(pdfUrl: api.artifactUrl(job.jobId, 'pdf')),
              ),
            ),
            const SizedBox(height: 24),

            // MIDI player section
            const Align(
              alignment: Alignment.centerLeft,
              child: Text(
                'Listen',
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.w700,
                  color: OhSheetColors.darkText,
                ),
              ),
            ),
            const SizedBox(height: 8),
            Container(
              height: 250,
              width: double.infinity,
              decoration: BoxDecoration(
                color: Colors.white,
                border: Border.all(color: Colors.grey.shade300),
                borderRadius: BorderRadius.circular(12),
              ),
              clipBehavior: Clip.antiAlias,
              child: MidiPlayerWidget(midiUrl: api.artifactUrl(job.jobId, 'midi')),
            ),
            const SizedBox(height: 24),

            // Download buttons
            Row(
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
            ),
            const SizedBox(height: 32),

            // Transcribe another
            SizedBox(
              width: double.infinity,
              child: OutlinedButton.icon(
                onPressed: () => Navigator.of(context).popUntil((r) => r.isFirst),
                icon: const Icon(Icons.refresh),
                label: const Text('Transcribe another song'),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
