// Screen tests for part A (sign in, create account, home). They use a fake login service,
// so they need no internet and no Supabase. Run: flutter test
import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:pest_assessment/config.dart';
import 'package:pest_assessment/main.dart';
import 'package:pest_assessment/services/auth_service.dart';
import 'package:pest_assessment/strings.dart';

class FakeAuth implements AuthService {
  FakeAuth({this.confirmEmail = false});
  final bool confirmEmail; // like Supabase's "Confirm email" setting
  final _changes = StreamController<bool>.broadcast();
  final users = <String, ({String password, String name, int barangay})>{};
  String? current;

  @override
  bool get isSignedIn => current != null;
  @override
  Stream<bool> get signedInChanges => _changes.stream;

  @override
  Future<void> signIn({required String email, required String password}) async {
    final u = users[email.trim()];
    if (u == null || u.password != password) throw const AuthFailure(S.wrongLogin);
    current = email.trim();
    _changes.add(true);
  }

  @override
  Future<bool> signUp({required String fullName, required int barangayId, required String email, required String password}) async {
    if (users.containsKey(email)) throw const AuthFailure(S.emailTaken);
    users[email] = (password: password, name: fullName, barangay: barangayId);
    if (confirmEmail) return false;
    current = email;
    _changes.add(true);
    return true;
  }

  @override
  Future<void> signOut() async {
    current = null;
    _changes.add(false);
  }

  @override
  Future<List<Barangay>> barangays() async => const [Barangay(1, 'Antonio Luna'), Barangay(13, 'La Union')];

  @override
  Future<Profile> profile() async {
    final u = users[current];
    return Profile(fullName: u?.name, barangay: u?.barangay == 13 ? 'La Union' : 'Antonio Luna');
  }
}

Future<void> pumpApp(WidgetTester tester, FakeAuth auth) async {
  tester.view.physicalSize = const Size(1080, 2400);
  tester.view.devicePixelRatio = 2.5;
  addTearDown(tester.view.reset);
  await tester.pumpWidget(CornPestApp(auth: auth));
  await tester.pumpAndSettle();
}

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

  testWidgets('starts on Sign in when signed out', (tester) async {
    await pumpApp(tester, FakeAuth());
    expect(find.text(S.appName), findsOneWidget);
    expect(find.widgetWithText(FilledButton, S.signIn), findsOneWidget);
  });

  testWidgets('empty or wrong sign-in shows clear messages', (tester) async {
    await pumpApp(tester, FakeAuth());
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
    await pumpApp(tester, auth);
    await tester.tap(find.text(S.createAccount));
    await tester.pumpAndSettle();
    await fillSignUp(tester);

    expect(auth.users['juan@example.com']?.barangay, 13);
    expect(find.text(S.hello('Juan Dela Cruz')), findsOneWidget);
    expect(find.text(S.fromBarangay('La Union')), findsOneWidget);
    expect(find.text(S.scanPlant), findsOneWidget);

    await tester.tap(find.byTooltip(S.signOut));
    await tester.pumpAndSettle();
    await tester.tap(find.widgetWithText(FilledButton, S.signOut));
    await tester.pumpAndSettle();
    expect(find.widgetWithText(FilledButton, S.signIn), findsOneWidget);

    // ... and signing in again works.
    await tester.enterText(find.byKey(const Key('email')), 'juan@example.com');
    await tester.enterText(find.byKey(const Key('password')), 'secret123');
    await tester.tap(find.widgetWithText(FilledButton, S.signIn));
    await tester.pumpAndSettle();
    expect(find.text(S.hello('Juan Dela Cruz')), findsOneWidget);
  });

  testWidgets('create account checks the form', (tester) async {
    await pumpApp(tester, FakeAuth());
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

  testWidgets('when email confirmation is on, explains it and returns to Sign in', (tester) async {
    await pumpApp(tester, FakeAuth(confirmEmail: true));
    await tester.tap(find.text(S.createAccount));
    await tester.pumpAndSettle();
    await fillSignUp(tester);
    expect(find.text(S.checkEmailTitle), findsOneWidget);
    await tester.tap(find.text(S.ok));
    await tester.pumpAndSettle();
    expect(find.widgetWithText(FilledButton, S.signIn), findsOneWidget);
  });

  testWidgets('an email that already has an account gets a clear message', (tester) async {
    final auth = FakeAuth()..users['juan@example.com'] = (password: 'x', name: 'Juan', barangay: 1);
    await pumpApp(tester, auth);
    await tester.tap(find.text(S.createAccount));
    await tester.pumpAndSettle();
    await fillSignUp(tester);
    expect(find.text(S.emailTaken), findsOneWidget);
  });
}
