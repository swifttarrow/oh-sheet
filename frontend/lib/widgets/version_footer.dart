import 'package:flutter/material.dart';

import '../theme.dart';

const appVersion = String.fromEnvironment('APP_VERSION', defaultValue: 'dev');

class VersionFooter extends StatelessWidget {
  const VersionFooter({super.key});

  @override
  Widget build(BuildContext context) {
    return Text(
      'v$appVersion',
      style: TextStyle(
        fontSize: 11,
        color: OhSheetColors.mutedText.withValues(alpha: 0.5),
        fontWeight: FontWeight.w500,
      ),
    );
  }
}
