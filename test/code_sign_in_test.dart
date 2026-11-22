// Sign in with a 6-digit email code (no password), and the one-time "About you" screen.
// The fake's correct code is FakeAuth.goodCode.
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:pest_assessment/strings.dart';

import 'fakes.dart';

Future<void> sendCodeTo(WidgetTester tester, String email) async {
  await tester.enterText(find.byKey(const Key('email')), email);
  await tester.tap(find.widgetWithText(FilledButton, S.sendCode));
  await tester.pumpAndSettle();
}

Future<void> enterCode(WidgetTester tester, String code) async {
  await tester.enterText(find.byKey(const Key('code')), code);
  await tester.tap(find.widgetWithText(FilledButton, S.signIn));
  await tester.pumpAndSettle();
}

void main() {
  testWidgets('new user: email -> code -> About you (name, barangay) -> Home', (tester) async {
    final auth = FakeAuth();
    await pumpCodeApp(tester, await makeServices(auth: auth));
    await sendCodeTo(tester, 'maria@example.com');
    expect(auth.codesSent, ['maria@example.com']);
    expect(find.text(S.codeSentTo('maria@example.com')), findsOneWidget);

    await enterCode(tester, FakeAuth.goodCode);
    expect(find.text(S.aboutYou), findsOneWidget);

    await tester.tap(find.widgetWithText(FilledButton, S.save));
    await tester.pumpAndSettle();
    expect(find.text(S.enterName), findsOneWidget);
    expect(find.text(S.chooseBarangay), findsOneWidget);

    await tester.enterText(find.byKey(const Key('fullName')), 'Maria Santos');
    await tester.tap(find.byKey(const Key('barangay')));
    await tester.pumpAndSettle();
    await tester.tap(find.text('La Union').last);
    await tester.pumpAndSettle();
    await tester.tap(find.widgetWithText(FilledButton, S.save));
    await tester.pumpAndSettle();
    expect(find.text(S.hello('Maria Santos')), findsOneWidget);
    expect(find.text(S.fromBarangay('La Union')), findsOneWidget);
    expect(auth.users['maria@example.com']?.barangay, 13);
    await tester.pump(const Duration(seconds: 61)); // let the resend timer finish
  });

  testWidgets('existing user with a profile goes straight to Home', (tester) async {
    final auth = FakeAuth()..users['juan@example.com'] = (password: 'x', name: 'Juan', barangay: 13);
    await pumpCodeApp(tester, await makeServices(auth: auth));
    await sendCodeTo(tester, 'juan@example.com');
    await enterCode(tester, FakeAuth.goodCode);
    expect(find.text(S.aboutYou), findsNothing);
    expect(find.text(S.hello('Juan')), findsOneWidget);
    await tester.pump(const Duration(seconds: 61));
  });

  testWidgets('wrong or missing code: clear messages; can change email; resend waits 60 s', (tester) async {
    final auth = FakeAuth();
    await pumpCodeApp(tester, await makeServices(auth: auth));
    await tester.tap(find.widgetWithText(FilledButton, S.sendCode));
    await tester.pumpAndSettle();
    expect(find.text(S.enterEmail), findsOneWidget);

    await tester.enterText(find.byKey(const Key('email')), 'ana@example.com');
    await tester.tap(find.widgetWithText(FilledButton, S.sendCode));
    await tester.pump(); // code sent
    await tester.pump();
    expect(find.text(S.resendIn(60)), findsOneWidget); // must wait before asking again
    await tester.pump(const Duration(seconds: 61));
    await enterCode(tester, '12');
    expect(find.text(S.enterCode), findsOneWidget);
    await enterCode(tester, '999999');
    expect(find.text(S.codeWrong), findsOneWidget);

    await tester.tap(find.text(S.resendCode));
    await tester.pumpAndSettle();
    expect(auth.codesSent, ['ana@example.com', 'ana@example.com']);

    await tester.tap(find.text(S.changeEmail));
    await tester.pumpAndSettle();
    expect(find.widgetWithText(FilledButton, S.sendCode), findsOneWidget);
    await tester.tap(find.text(S.usePassword)); // back to the login form
    await tester.pumpAndSettle();
    expect(find.text(S.useCode), findsOneWidget);
    await tester.pump(const Duration(seconds: 61));
  });

  testWidgets('password sign-in still works and lands on Home', (tester) async {
    final auth = FakeAuth()..users['juan@example.com'] = (password: 'secret123', name: 'Juan', barangay: 13);
    await pumpPasswordApp(tester, await makeServices(auth: auth));
    await tester.enterText(find.byKey(const Key('email')), 'juan@example.com');
    await tester.enterText(find.byKey(const Key('password')), 'secret123');
    await tester.tap(find.widgetWithText(FilledButton, S.signIn));
    await tester.pumpAndSettle();
    expect(find.text(S.hello('Juan')), findsOneWidget);
  });
}
