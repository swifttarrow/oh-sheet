import 'package:flutter/material.dart';

import '../theme.dart';
import 'sticker_widgets.dart';

class LegalDisclaimerDialog extends StatelessWidget {
  const LegalDisclaimerDialog({super.key});

  static const String titleText = 'Responsible Use Notice';
  static const String introText =
      'Use Oh Sheet only with content you are legally allowed to access, upload, convert, download, and share.';
  static const List<String> bulletPoints = [
    'You are responsible for complying with copyright law, licenses, platform terms, and any permissions required for the audio, video, MIDI, sheet music, or other material you use with this app.',
    'Do not use Oh Sheet to infringe intellectual property rights, misuse third-party content, bypass restrictions, or otherwise violate applicable law or terms of service.',
    'AI-generated transcriptions, arrangements, and sheet music may contain errors or omissions, so review all outputs before relying on or distributing them.',
  ];
  static const String acknowledgementText =
      'By continuing, you acknowledge that you are solely responsible for your use of Oh Sheet and any content you submit or generate. Oh Sheet is provided as-is, and Oh Sheet takes no responsibility for improper, unauthorized, or unlawful use.';
  static const String noteText =
      'If you are unsure whether you have permission to use specific content, do not upload it.';

  static Future<void> show(BuildContext context) {
    return showDialog<void>(
      context: context,
      barrierDismissible: false,
      builder: (_) => const LegalDisclaimerDialog(),
    );
  }

  @override
  Widget build(BuildContext context) {
    return PopScope(
      canPop: false,
      child: Dialog(
        backgroundColor: Colors.transparent,
        insetPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 24),
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 560),
          child: DecoratedBox(
            decoration: BoxDecoration(
              gradient: const LinearGradient(
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
                colors: [
                  Color(0xFFFFFCF7),
                  Color(0xFFF6FFFD),
                ],
              ),
              borderRadius: BorderRadius.circular(OhSheetStickerStyle.radiusLg),
              border: Border.all(
                color: OhSheetColors.inkStroke,
                width: OhSheetStickerStyle.borderWidth,
              ),
              boxShadow: OhSheetStickerStyle.stickerShadows,
            ),
            child: SingleChildScrollView(
              padding: const EdgeInsets.fromLTRB(20, 20, 20, 20),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                    decoration: BoxDecoration(
                      color: const Color(0xFFFFE8B7),
                      borderRadius: BorderRadius.circular(999),
                      border: Border.all(
                        color: OhSheetColors.inkStroke,
                        width: 2,
                      ),
                    ),
                    child: const Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(
                          Icons.gavel_rounded,
                          color: OhSheetColors.orange,
                          size: 18,
                        ),
                        SizedBox(width: 8),
                        Text(
                          'Legal',
                          style: TextStyle(
                            fontWeight: FontWeight.w800,
                            letterSpacing: 0.4,
                            color: OhSheetColors.darkText,
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 16),
                  const Text(
                    titleText,
                    style: TextStyle(
                      fontSize: 28,
                      height: 1.05,
                      fontWeight: FontWeight.w900,
                      color: OhSheetColors.darkText,
                    ),
                  ),
                  const SizedBox(height: 12),
                  const Text(
                    introText,
                    style: TextStyle(
                      fontSize: 14,
                      height: 1.5,
                      color: OhSheetColors.mutedText,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  const SizedBox(height: 18),
                  for (final bullet in bulletPoints) ...[
                    _LegalBullet(text: bullet),
                    const SizedBox(height: 12),
                  ],
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: const Color(0xFFFFE8EF),
                      borderRadius: BorderRadius.circular(20),
                      border: Border.all(
                        color: OhSheetColors.inkStroke,
                        width: 2.5,
                      ),
                    ),
                    child: const Text(
                      acknowledgementText,
                      style: TextStyle(
                        fontSize: 14,
                        height: 1.5,
                        color: OhSheetColors.darkText,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ),
                  const SizedBox(height: 12),
                  const Text(
                    noteText,
                    style: TextStyle(
                      fontSize: 13,
                      height: 1.5,
                      color: OhSheetColors.mutedText,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  const SizedBox(height: 20),
                  SizedBox(
                    width: double.infinity,
                    child: FilledButton.icon(
                      onPressed: () => Navigator.of(context).pop(),
                      icon: const Icon(Icons.verified_user_rounded),
                      label: const Text('Continue responsibly'),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class _LegalBullet extends StatelessWidget {
  const _LegalBullet({required this.text});

  final String text;

  @override
  Widget build(BuildContext context) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Container(
          width: 26,
          height: 26,
          margin: const EdgeInsets.only(top: 2),
          decoration: BoxDecoration(
            color: OhSheetColors.teal.withValues(alpha: 0.16),
            shape: BoxShape.circle,
            border: Border.all(
              color: OhSheetColors.inkStroke,
              width: 1.5,
            ),
          ),
          child: const Icon(
            Icons.check_rounded,
            size: 16,
            color: OhSheetColors.teal,
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: Text(
            text,
            style: const TextStyle(
              fontSize: 14,
              height: 1.5,
              color: OhSheetColors.darkText,
              fontWeight: FontWeight.w600,
            ),
          ),
        ),
      ],
    );
  }
}
