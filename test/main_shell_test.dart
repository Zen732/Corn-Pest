// The signed-in app: bottom bar (Home, History, camera, Settings), the Home page, Settings.
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:pest_assessment/services/history_store.dart';
import 'package:pest_assessment/services/scan_service.dart';
import 'package:pest_assessment/strings.dart';

import 'fakes.dart';

const pest = ScanResult(outcome: 'pest', pest: 'fall_armyworm', pestConfidence: 0.84);
const healthy = ScanResult(outcome: 'healthy', severity: 'Healthy');

void main() {
  testWidgets('Home: greeting, numbers, recent scans, See all opens History', (tester) async {
    final remote = FakeRemote()..online = false;
    final history = HistoryStore(storage: MemoryStorage(), remote: remote, autoSync: false);
    await history.save(healthy, tinyPng, at: DateTime(2026, 11, 1));
    await history.save(pest, tinyPng, at: DateTime(2026, 11, 2));
    await pumpApp(tester, await makeServices(auth: FakeAuth.signedIn(), history: history));

    expect(find.text(S.hello('Juan Dela Cruz')), findsOneWidget);
    expect(find.text(S.fromBarangay('La Union')), findsOneWidget);
    expect(find.text(S.statScans), findsOneWidget);
    expect(find.text('2'), findsWidgets); // 2 scans, 2 waiting
    expect(find.text('1'), findsOneWidget); // 1 pest
    expect(find.text(S.summary('pest', 'fall_armyworm', null)), findsOneWidget);

    await tester.tap(find.text(S.seeAll));
    await tester.pumpAndSettle();
    expect(find.text(S.waitingCount(2)), findsOneWidget); // the History tab
  });

  testWidgets('bottom bar: icons switch tabs; camera opens Scan', (tester) async {
    await pumpApp(tester, await makeServices(auth: FakeAuth.signedIn()));
    expect(find.byTooltip(S.home), findsOneWidget);
    expect(find.byTooltip(S.history), findsOneWidget);
    expect(find.byTooltip(S.settings), findsOneWidget);

    await tester.tap(find.byTooltip(S.settings));
    await tester.pumpAndSettle();
    expect(find.text('juan@example.com'), findsOneWidget);
    await tester.tap(find.byTooltip(S.history));
    await tester.pumpAndSettle();
    expect(find.text(S.historyEmpty), findsOneWidget);
    await tester.tap(find.byTooltip(S.home));
    await tester.pumpAndSettle();
    expect(find.text(S.noScansYet), findsOneWidget);

    await tester.tap(find.byTooltip(S.scanPlant));
    await tester.pumpAndSettle();
    expect(find.text(S.takePhoto), findsOneWidget);
  });

  testWidgets('Settings: shows info; edit name and barangay updates Home', (tester) async {
    final auth = FakeAuth.signedIn();
    await pumpApp(tester, await makeServices(auth: auth));
    await tester.tap(find.byTooltip(S.settings));
    await tester.pumpAndSettle();
    expect(find.text('Juan Dela Cruz'), findsOneWidget);
    expect(find.text('La Union'), findsOneWidget);

    await tester.tap(find.text(S.editProfile));
    await tester.pumpAndSettle();
    await tester.enterText(find.byKey(const Key('fullName')), 'Juan D. Cruz');
    await tester.tap(find.byKey(const Key('barangay')));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Antonio Luna').last);
    await tester.pumpAndSettle();
    await tester.tap(find.widgetWithText(FilledButton, S.save));
    await tester.pumpAndSettle();
    expect(auth.users['juan@example.com']?.barangay, 1);
    expect(find.text('Juan D. Cruz'), findsOneWidget); // Settings refreshed

    await tester.tap(find.byTooltip(S.home));
    await tester.pumpAndSettle();
    expect(find.text(S.hello('Juan D. Cruz')), findsOneWidget);
    expect(find.text(S.fromBarangay('Antonio Luna')), findsOneWidget);
  });

  testWidgets('Settings: change password checks the boxes and saves', (tester) async {
    final auth = FakeAuth.signedIn();
    await pumpApp(tester, await makeServices(auth: auth));
    await tester.tap(find.byTooltip(S.settings));
    await tester.pumpAndSettle();
    await tester.tap(find.text(S.changePassword));
    await tester.pumpAndSettle();
    await tester.enterText(find.byKey(const Key('newPassword')), 'abc');
    await tester.tap(find.widgetWithText(FilledButton, S.save));
    await tester.pumpAndSettle();
    expect(find.text(S.passwordTooShort), findsOneWidget);
    await tester.enterText(find.byKey(const Key('newPassword')), 'newsecret1');
    await tester.enterText(find.byKey(const Key('repeatPassword')), 'newsecret1');
    await tester.tap(find.widgetWithText(FilledButton, S.save));
    await tester.pumpAndSettle();
    expect(find.text(S.passwordChanged), findsOneWidget);
    expect(auth.users['juan@example.com']?.password, 'newsecret1');
  });

  testWidgets('Settings: Upload now sends waiting scans; Sign out returns to the login form', (tester) async {
    final remote = FakeRemote()..online = false;
    final history = HistoryStore(storage: MemoryStorage(), remote: remote, autoSync: false);
    await history.save(pest, tinyPng);
    await pumpApp(tester, await makeServices(auth: FakeAuth.signedIn(), history: history));
    await tester.tap(find.byTooltip(S.settings));
    await tester.pumpAndSettle();
    expect(find.text(S.waitingCount(1)), findsOneWidget);

    remote.online = true;
    await tester.tap(find.text(S.uploadNow));
    await tester.pumpAndSettle();
    expect(remote.rows, hasLength(1));
    expect(find.text(S.allUploaded), findsWidgets);

    await signOutViaSettings(tester);
    expect(find.widgetWithText(FilledButton, S.signIn), findsOneWidget);
  });
}
