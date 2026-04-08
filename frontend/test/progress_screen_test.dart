// TDD: Tests for the restyled progress screen with mascot and stage badges.
import 'package:flutter_test/flutter_test.dart';

import 'package:ohsheet_app/screens/progress_screen.dart';

void main() {
  group('friendlyStageName', () {
    test('maps ingest to friendly name', () {
      expect(friendlyStageName('ingest'), 'Preparing');
    });

    test('maps transcribe to friendly name', () {
      expect(friendlyStageName('transcribe'), 'Transcribing');
    });

    test('maps arrange to friendly name', () {
      expect(friendlyStageName('arrange'), 'Arranging');
    });

    test('maps humanize to friendly name', () {
      expect(friendlyStageName('humanize'), 'Humanizing');
    });

    test('maps engrave to friendly name', () {
      expect(friendlyStageName('engrave'), 'Engraving');
    });

    test('returns raw name for unknown stage', () {
      expect(friendlyStageName('unknown'), 'unknown');
    });
  });

  group('mascotAssetForStage', () {
    test('returns ingest mascot', () {
      expect(mascotAssetForStage('ingest'), 'assets/mascots/mascot-progress-ingest.png');
    });

    test('returns transcribe mascot', () {
      expect(mascotAssetForStage('transcribe'), 'assets/mascots/mascot-progress-transcribe.png');
    });

    test('returns arrange mascot for arrange and humanize', () {
      expect(mascotAssetForStage('arrange'), 'assets/mascots/mascot-progress-arrange.png');
      expect(mascotAssetForStage('humanize'), 'assets/mascots/mascot-progress-arrange.png');
    });

    test('returns engrave mascot', () {
      expect(mascotAssetForStage('engrave'), 'assets/mascots/mascot-progress-engrave.png');
    });
  });

  group('pipelineTips', () {
    test('contains at least 3 tips', () {
      expect(pipelineTips.length, greaterThanOrEqualTo(3));
    });

    test('tips are non-empty strings', () {
      for (final tip in pipelineTips) {
        expect(tip.isNotEmpty, isTrue);
      }
    });
  });
}
