import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:ohsheet_app/widgets/version_footer.dart';

void main() {
  testWidgets('VersionFooter displays version text', (tester) async {
    await tester.pumpWidget(
      const MaterialApp(
        home: Scaffold(
          body: VersionFooter(),
        ),
      ),
    );

    // Default value when APP_VERSION is not defined at compile time
    expect(find.text('vdev'), findsOneWidget);
  });
}
