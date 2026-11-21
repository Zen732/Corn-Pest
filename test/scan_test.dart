// Screen tests for part B (photo tips, taking a photo, the result, saving to History).
// Fakes stand in for the camera, the backend and Supabase. Run: flutter test
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:pest_assessment/services/history_store.dart';
import 'package:pest_assessment/services/scan_service.dart';
import 'package:pest_assessment/strings.dart';

import 'fakes.dart';

const pestResult = ScanResult(
  outcome: 'pest',
  pest: 'fall_armyworm',
  pestConfidence: 0.84,
  severity: 'Moderate',
  damagePercent: 23.4,
  insectVisible: true,
  cornConfidence: 0.98,
  modelVersion: '2026-11-14',
);

Future<void> openScan(WidgetTester tester) async {
  await tester.tap(find.text(S.scanPlant));
  await tester.pumpAndSettle();
}

Future<void> takePhoto(WidgetTester tester) async {
  await tester.tap(find.text(S.takePhoto));
  await tester.pumpAndSettle();
}

void main() {
  testWidgets('tips show until "Don\'t show again" is ticked', (tester) async {
    final services = await makeServices(auth: FakeAuth.signedIn());
    await pumpApp(tester, services);
    await openScan(tester);
    expect(find.text(S.photoTips), findsOneWidget);
    expect(find.text(S.tips.first), findsOneWidget);

    await tester.tap(find.byKey(const Key('hideTips')));
    await tester.pumpAndSettle();
    expect(services.tips.hideTips, isTrue);

    await tester.pageBack();
    await tester.pumpAndSettle();
    await openScan(tester);
    expect(find.text(S.photoTips), findsNothing);
    await tester.tap(find.text(S.showTips));
    await tester.pumpAndSettle();
    expect(find.text(S.photoTips), findsOneWidget);
  });

  testWidgets('pest result: name, scientific name, how sure, severity; saved to History', (tester) async {
    final history = HistoryStore(storage: MemoryStorage(), remote: FakeRemote());
    final photos = FakePhotos();
    await pumpApp(tester, await makeServices(auth: FakeAuth.signedIn(), api: FakeApi([pestResult]), history: history, photos: photos));
    await openScan(tester);
    await takePhoto(tester);

    expect(photos.lastCamera, isTrue);
    expect(find.text('Fall armyworm'), findsOneWidget);
    expect(find.text('Spodoptera frugiperda'), findsOneWidget);
    expect(find.text(S.howSure(0.84)), findsOneWidget);
    expect(find.text(S.severityLine('Moderate', 23.4)), findsOneWidget);
    expect(find.text(S.saved), findsOneWidget);
    expect((await history.list()).single.result.pest, 'fall_armyworm');
  });

  testWidgets('pest with no plant in the photo: severity "not measured"', (tester) async {
    const result = ScanResult(outcome: 'pest', pest: 'corn_aphid', pestConfidence: 0.7, severity: 'Unclear');
    await pumpApp(tester, await makeServices(auth: FakeAuth.signedIn(), api: FakeApi([result])));
    await openScan(tester);
    await takePhoto(tester);
    expect(find.text('Corn aphid'), findsOneWidget);
    expect(find.text(S.severityNotMeasured), findsOneWidget);
  });

  testWidgets('healthy and damage-only results, both saved', (tester) async {
    final history = HistoryStore(storage: MemoryStorage(), remote: FakeRemote());
    final api = FakeApi(const [
      ScanResult(outcome: 'healthy', severity: 'Healthy'),
      ScanResult(outcome: 'severity_only', severity: 'High', damagePercent: 52),
    ]);
    await pumpApp(tester, await makeServices(auth: FakeAuth.signedIn(), api: api, history: history));
    await openScan(tester);
    await takePhoto(tester);
    expect(find.text(S.healthyTitle), findsOneWidget);
    expect(find.text(S.healthyBody), findsOneWidget);

    await tester.tap(find.text(S.scanAnother));
    await tester.pumpAndSettle();
    await takePhoto(tester);
    expect(find.text(S.damageTitle), findsOneWidget);
    expect(find.text(S.severityLine('High', 52)), findsOneWidget);
    expect(find.text(S.noInsectSeen), findsOneWidget);
    expect((await history.list()).map((s) => s.result.outcome).toSet(), {'healthy', 'severity_only'});
  });

  testWidgets('not corn: message and retake, nothing saved', (tester) async {
    final history = HistoryStore(storage: MemoryStorage(), remote: FakeRemote());
    await pumpApp(tester, await makeServices(
        auth: FakeAuth.signedIn(), api: FakeApi(const [ScanResult(outcome: 'not_corn')]), history: history));
    await openScan(tester);
    await tester.tap(find.text(S.chooseFromGallery));
    await tester.pumpAndSettle();
    expect(find.text(S.notCornTitle), findsOneWidget);
    expect(find.text(S.retakePhoto), findsOneWidget);
    expect(find.text(S.saved), findsNothing);
    expect(await history.list(), isEmpty);

    await tester.tap(find.text(S.retakePhoto));
    await tester.pumpAndSettle();
    expect(find.text(S.takePhoto), findsOneWidget);
  });

  testWidgets('server problem: clear message, then Try again works', (tester) async {
    final api = FakeApi(const [ScanFailure(S.serverUnreachable), pestResult]);
    await pumpApp(tester, await makeServices(auth: FakeAuth.signedIn(), api: api));
    await openScan(tester);
    await takePhoto(tester);
    expect(find.text(S.serverUnreachable), findsOneWidget);

    await tester.tap(find.text(S.tryAgain));
    await tester.pumpAndSettle();
    expect(find.text('Fall armyworm'), findsOneWidget);
    expect(api.calls, 2);
  });

  testWidgets('saving fails (storage full): says so, and Try again saves it', (tester) async {
    final storage = FlakyStorage()..failures = 1;
    final history = HistoryStore(storage: storage, remote: FakeRemote());
    await pumpApp(tester, await makeServices(auth: FakeAuth.signedIn(), api: FakeApi([pestResult]), history: history));
    await openScan(tester);
    await takePhoto(tester);
    expect(find.text(S.saveFailed), findsOneWidget);
    await tester.tap(find.text(S.tryAgain));
    await tester.pumpAndSettle();
    expect(find.text(S.saved), findsOneWidget);
    expect(await history.list(), hasLength(1));
  });

  testWidgets('cancelling the camera stays on the scan screen; Done goes Home', (tester) async {
    final api = FakeApi([pestResult]);
    final photos = FakePhotos()..next = null;
    await pumpApp(tester, await makeServices(auth: FakeAuth.signedIn(), api: api, photos: photos));
    await openScan(tester);
    await takePhoto(tester);
    expect(api.calls, 0);
    expect(find.text(S.takePhoto), findsOneWidget);

    photos.next = tinyPng;
    await takePhoto(tester);
    await tester.tap(find.text(S.done));
    await tester.pumpAndSettle();
    expect(find.text(S.hello('Juan Dela Cruz')), findsOneWidget);
  });

  test('the backend answer is read correctly', () {
    final r = ScanResult.fromJson({
      'outcome': 'pest', 'pest': 'fall_armyworm', 'pest_confidence': 0.84, 'severity': 'Unclear',
      'damage_percent': null, 'insect_visible': true, 'insect_confidence': 0.97, 'healthy': null,
      'health_confidence': null, 'corn_confidence': 0.98, 'model_version': '2026-11-14',
    });
    expect(r.savable, isTrue);
    expect(r.pestConfidence, 0.84);
    expect(r.damagePercent, isNull);
    expect(r.modelVersion, '2026-11-14');
    expect(const ScanResult(outcome: 'not_corn').savable, isFalse);
    expect(const ScanResult(outcome: 'unclear').savable, isFalse);
  });
}
