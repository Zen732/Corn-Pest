// Screen tests for part A (sign in, create account, home). They use a fake login service,
// so they need no internet and no Supabase. Run: flutter test
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:pest_assessment/config.dart';
import 'package:pest_assessment/strings.dart';

import 'fakes.dart';

Future<void> fillSignUp(WidgetTester tester, {String email = 'juan@example.com'}) async {
  await tester.enterText(find.byKey(const Key('fullName')), 'Juan Dela Cruz');
  await tester.tap(find.byKey(const Key('barangay')));
  await tester.pumpAndSettle();
  await tester.tap(find.text('La Union').last);
  await tester.pumpAndSettle();
  await tester.enterText(find.byKey(const Key('email')), email);
  await tester.enterText(find.byKey(const Key('password')), 'secret123');
  await tester.enterText(find.byKey(const Key('repeat')), 'secret123');
  await tester.tap(find.widgetWithText(FilledButton, S.createAccount));
  await tester.pumpAndSettle();
}

void main() {
  test('Supabase URL works with or without /rest/v1/ at the end', () {
    expect(Config.normalizeSupabaseUrl('https://abc.supabase.co/rest/v1/'), 'https://abc.supabase.co');
    expect(Config.normalizeSupabaseUrl(' https://abc.supabase.co/ '), 'https://abc.supabase.co');
    expect(Config.normalizeSupabaseUrl('https://abc.supabase.co'), 'https://abc.supabase.co');
  });

  testWidgets('starts on the login form, with code and create-account options', (tester) async {
    await pumpApp(tester, await makeServices(auth: FakeAuth()));
    expect(find.text(S.appName), findsOneWidget);
    expect(find.widgetWithText(FilledButton, S.signIn), findsOneWidget);
    expect(find.text(S.useCode), findsOneWidget);
    expect(find.text(S.createAccount), findsOneWidget);
  });

  testWidgets('empty or wrong sign-in shows clear messages', (tester) async {
    await pumpPasswordApp(tester, await makeServices(auth: FakeAuth()));
    await tester.tap(find.widgetWithText(FilledButton, S.signIn));
    await tester.pumpAndSettle();
    expect(find.text(S.enterEmail), findsOneWidget);
    expect(find.text(S.enterPassword), findsOneWidget);

    await tester.enterText(find.byKey(const Key('email')), 'nobody@example.com');
    await tester.enterText(find.byKey(const Key('password')), 'wrong');
    await tester.tap(find.widgetWithText(FilledButton, S.signIn));
    await tester.pumpAndSettle();
    expect(find.text(S.wrongLogin), findsOneWidget);
  });

  testWidgets('create account -> Home greets by name and barangay; sign out -> Sign in', (tester) async {
    final auth = FakeAuth();
    await pumpPasswordApp(tester, await makeServices(auth: auth));
    await tester.tap(find.text(S.createAccount));
    await tester.pumpAndSettle();
    await fillSignUp(tester);

    expect(auth.users['juan@example.com']?.barangay, 13);
    expect(find.text(S.hello('Juan Dela Cruz')), findsOneWidget);
    expect(find.text(S.fromBarangay('La Union')), findsOneWidget);
    expect(find.byTooltip(S.scanPlant), findsOneWidget);

    await signOutViaSettings(tester);
    expect(find.widgetWithText(FilledButton, S.signIn), findsOneWidget);

    // ... and signing in again works.
    await tester.enterText(find.byKey(const Key('email')), 'juan@example.com');
    await tester.enterText(find.byKey(const Key('password')), 'secret123');
    await tester.tap(find.widgetWithText(FilledButton, S.signIn));
    await tester.pumpAndSettle();
    expect(find.text(S.hello('Juan Dela Cruz')), findsOneWidget);
  });

  testWidgets('eye icons show and hide the typed passwords', (tester) async {
    bool hidden(String key) => tester
        .widget<EditableText>(find.descendant(of: find.byKey(Key(key)), matching: find.byType(EditableText)))
        .obscureText;

    await pumpPasswordApp(tester, await makeServices(auth: FakeAuth()));
    expect(hidden('password'), isTrue);
    await tester.tap(find.byTooltip(S.showPassword));
    await tester.pump();
    expect(hidden('password'), isFalse);

    await tester.tap(find.text(S.createAccount));
    await tester.pumpAndSettle();
    expect(hidden('password'), isTrue);
    expect(hidden('repeat'), isTrue);
    await tester.tap(find.byKey(const Key('showPassword')));
    await tester.pump();
    expect(hidden('password'), isFalse);
    expect(hidden('repeat'), isTrue); // each box has its own eye
    await tester.tap(find.byKey(const Key('showRepeat')));
    await tester.pump();
    expect(hidden('repeat'), isFalse);
    await tester.tap(find.byKey(const Key('showRepeat')));
    await tester.pump();
    expect(hidden('repeat'), isTrue);
  });

  testWidgets('create account checks the form', (tester) async {
    await pumpPasswordApp(tester, await makeServices(auth: FakeAuth()));
    await tester.tap(find.text(S.createAccount));
    await tester.pumpAndSettle();
    await tester.enterText(find.byKey(const Key('password')), '123');
    await tester.enterText(find.byKey(const Key('repeat')), '456');
    await tester.tap(find.widgetWithText(FilledButton, S.createAccount));
    await tester.pumpAndSettle();
    expect(find.text(S.enterName), findsOneWidget);
    expect(find.text(S.chooseBarangay), findsOneWidget);
    expect(find.text(S.enterEmail), findsOneWidget);
    expect(find.text(S.passwordTooShort), findsOneWidget);
    expect(find.text(S.passwordsDiffer), findsOneWidget);
  });

  testWidgets('when email confirmation is on, asks for the emailed code, then Home', (tester) async {
    final auth = FakeAuth(confirmEmail: true);
    await pumpPasswordApp(tester, await makeServices(auth: auth));
    await tester.tap(find.text(S.createAccount));
    await tester.pumpAndSettle();
    await fillSignUp(tester);
    expect(find.byKey(const Key('code')), findsOneWidget); // no link: a code box
    await tester.enterText(find.byKey(const Key('code')), FakeAuth.goodCode);
    await tester.tap(find.widgetWithText(FilledButton, S.signIn));
    await tester.pumpAndSettle();
    expect(auth.unconfirmed, isEmpty);
    expect(find.byTooltip(S.scanPlant), findsOneWidget);
  });

  testWidgets('password sign-in on an unconfirmed account offers a code', (tester) async {
    final auth = FakeAuth()
      ..users['ana@example.com'] = (password: 'secret123', name: 'Ana', barangay: 1)
      ..unconfirmed.add('ana@example.com');
    await pumpPasswordApp(tester, await makeServices(auth: auth));
    await tester.enterText(find.byKey(const Key('email')), 'ana@example.com');
    await tester.enterText(find.byKey(const Key('password')), 'secret123');
    await tester.tap(find.widgetWithText(FilledButton, S.signIn));
    await tester.pumpAndSettle();
    expect(find.text(S.emailNotConfirmed), findsOneWidget);
    await tester.tap(find.text(S.confirmWithCode));
    await tester.pumpAndSettle();
    expect(auth.codesSent, ['ana@example.com']);
    await tester.enterText(find.byKey(const Key('code')), FakeAuth.goodCode);
    await tester.tap(find.widgetWithText(FilledButton, S.signIn));
    await tester.pumpAndSettle();
    expect(find.byTooltip(S.scanPlant), findsOneWidget);
  });

  testWidgets('an email that already has an account gets a clear message', (tester) async {
    final auth = FakeAuth()..users['juan@example.com'] = (password: 'x', name: 'Juan', barangay: 1);
    await pumpPasswordApp(tester, await makeServices(auth: auth));
    await tester.tap(find.text(S.createAccount));
    await tester.pumpAndSettle();
    await fillSignUp(tester);
    expect(find.text(S.emailTaken), findsOneWidget);
  });
}
