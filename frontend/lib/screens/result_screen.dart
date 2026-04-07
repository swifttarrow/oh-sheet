/// Result screen — shows job metadata + Download PDF / Download MIDI buttons.
///
/// Downloads use url_launcher to hand the artifact URL off to the OS so the
/// browser / system download manager handles streaming the bytes.
library;

import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';

import '../api/client.dart';
import '../api/models.dart';
import 'upload_screen.dart';

class ResultScreen extends StatelessWidget {
  const ResultScreen({super.key, required this.api, required this.job});
  final OhSheetApi api;
  final JobSummary job;

  Future<void> _download(BuildContext context, String kind) async {
    final url = api.artifactUrl(job.jobId, kind);
    final ok = await launchUrl(Uri.parse(url), mode: LaunchMode.externalApplication);
    if (!ok && context.mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Could not open $url')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final result = job.result;
    final title = job.title ?? 'Untitled';
    final artist = job.artist;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Done'),
        leading: IconButton(
          icon: const Icon(Icons.home),
          onPressed: () => Navigator.of(context).pushAndRemoveUntil(
            MaterialPageRoute(builder: (_) => UploadScreen(api: api)),
            (route) => false,
          ),
        ),
      ),
      body: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(title, style: Theme.of(context).textTheme.headlineSmall),
                    if (artist != null && artist.isNotEmpty)
                      Padding(
                        padding: const EdgeInsets.only(top: 4),
                        child: Text(artist, style: Theme.of(context).textTheme.titleMedium),
                      ),
                    const SizedBox(height: 8),
                    Text('Job ${job.jobId} · ${job.variant}',
                        style: Theme.of(context).textTheme.bodySmall),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 24),
            FilledButton.icon(
              onPressed: result == null ? null : () => _download(context, 'pdf'),
              icon: const Icon(Icons.picture_as_pdf),
              label: const Text('Download PDF'),
            ),
            const SizedBox(height: 12),
            FilledButton.icon(
              onPressed: result == null ? null : () => _download(context, 'midi'),
              icon: const Icon(Icons.music_note),
              label: const Text('Download MIDI'),
            ),
            const SizedBox(height: 12),
            OutlinedButton.icon(
              onPressed: result == null ? null : () => _download(context, 'musicxml'),
              icon: const Icon(Icons.notes),
              label: const Text('Download MusicXML'),
            ),
            const Spacer(),
            TextButton(
              onPressed: () => Navigator.of(context).pushAndRemoveUntil(
                MaterialPageRoute(builder: (_) => UploadScreen(api: api)),
                (route) => false,
              ),
              child: const Text('Transcribe another song'),
            ),
          ],
        ),
      ),
    );
  }
}
