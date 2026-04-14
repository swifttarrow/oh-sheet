// Widget tests for progress screen refine stage rendering (Plan 03-05).
// UX-03 (refine in display list + stage name + mascot arm), UX-04 (distinct
// "Refinement unavailable" badge on refine_skipped, not red-error).
// Default-path invariant: unrefined jobs still reach 1.0 progress (I-05).
//
// Every test in this file executes — there are no skip annotations and no
// skipped-test flags. RefineSkippedBadge is a PUBLIC widget (see Task 2) so
// tests pump it directly without needing to substitute JobEventStream.
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:ohsheet_app/screens/progress_screen.dart';
import 'package:ohsheet_app/theme.dart';

void main() {
  group('UX-03 — refine stage naming + mascot', () {
    test('friendlyStageName("refine") returns "Refining"', () {
      expect(friendlyStageName('refine'), 'Refining');
    });

    test('mascotAssetForStage("refine") returns an existing mascot asset', () {
      final asset = mascotAssetForStage('refine');
      expect(asset, endsWith('.svg'));
      expect(asset, isNot(contains('error')),
          reason: 'refine stage must not render the error mascot');
      expect(asset, startsWith('assets/mascots/'));
    });

    test('the refine stage name is user-visible, NOT the raw "refine" string', () {
      // If the switch falls through to its default "_ => stage" arm, the
      // raw "refine" would show. This test guards against accidentally
      // removing the arm later.
      expect(friendlyStageName('refine'), isNot('refine'));
    });
  });

  group('UX-03 — expectedStagesFor includes refine when enableRefine=true', () {
    test('expectedStagesFor(enableRefine: true) contains "refine" between arrange and engrave', () {
      final stages = expectedStagesFor(enableRefine: true);
      expect(stages, contains('refine'));
      final arrangeIdx = stages.indexOf('arrange');
      final refineIdx = stages.indexOf('refine');
      final engraveIdx = stages.indexOf('engrave');
      expect(arrangeIdx, lessThan(refineIdx),
          reason: 'refine must slot after arrange');
      expect(refineIdx, lessThan(engraveIdx),
          reason: 'refine must slot before engrave');
    });

    test('expectedStagesFor(enableRefine: false) does NOT contain "refine"', () {
      final stages = expectedStagesFor(enableRefine: false);
      expect(stages, isNot(contains('refine')));
    });
  });

  group('UX-04 — refine_skipped badge rendering', () {
    testWidgets('renders "Refinement unavailable" and does NOT render error state',
        (tester) async {
      // Pump RefineSkippedBadge directly — it's public as of Task 2.
      await tester.pumpWidget(
        const MaterialApp(
          home: Scaffold(
            body: RefineSkippedBadge(),
          ),
        ),
      );
      expect(find.text('Refinement unavailable'), findsOneWidget);

      // Distinct from red error styling: find the badge and inspect its
      // decoration color.
      final container = tester.widget<Container>(
        find
            .ancestor(
              of: find.text('Refinement unavailable'),
              matching: find.byType(Container),
            )
            .first,
      );
      final decoration = container.decoration as BoxDecoration;
      expect(decoration.color, isNot(OhSheetColors.error),
          reason: 'UX-04: must NOT be the red error color');
      expect(decoration.color, isNot(OhSheetColors.success),
          reason: 'UX-04: must NOT be confused with the green success color');
    });
  });

  group('I-05 — unrefined job progress reaches 1.0, not 0.83', () {
    test('default-path 5-stage completion yields computeProgress == 1.0', () {
      // The key invariant: for an UNREFINED job (the default, dominant
      // post-Phase-3 path), completion of all 5 base stages yields
      // fraction == 1.0 — NOT 5/6 ≈ 0.83 (which would happen if the
      // denominator were forced to 6 by mutating kPipelineStages).
      final completed = <String>{
        'ingest',
        'transcribe',
        'arrange',
        'humanize',
        'engrave',
      };
      final progress = computeProgress(
        completed: completed,
        enableRefine: false,
      );
      expect(progress, 1.0,
          reason: 'default-path 5-stage job must reach 100% progress');
      expect(progress, isNot(closeTo(5 / 6, 0.01)),
          reason: 'must NOT be 0.83 — the regression this test guards against');
    });

    test('refined-path 5-of-6-stage completion yields computeProgress < 1.0', () {
      // Symmetric check: when enable_refine=true but only the 5 base
      // stages have completed (not refine), progress should be <1.0 —
      // the denominator correctly adapted to 6.
      final completed = <String>{
        'ingest',
        'transcribe',
        'arrange',
        'humanize',
        'engrave',
      };
      final progress = computeProgress(
        completed: completed,
        enableRefine: true,
      );
      expect(progress, lessThan(1.0),
          reason: 'refined job with refine not yet completed must be <1.0');
    });

    test('refined-path with all 6 stages complete yields computeProgress == 1.0', () {
      final completed = <String>{
        'ingest',
        'transcribe',
        'arrange',
        'humanize',
        'refine',
        'engrave',
      };
      final progress = computeProgress(
        completed: completed,
        enableRefine: true,
      );
      expect(progress, 1.0,
          reason: 'refined job with all stages complete must reach 100%');
    });
  });
}
