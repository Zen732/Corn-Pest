// Fakes for the screen tests: no internet, camera, backend or Supabase needed.
import 'dart:async';
import 'dart:convert';
import 'dart:math';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:pest_assessment/main.dart';
import 'package:pest_assessment/services/app_services.dart';
import 'package:pest_assessment/services/auth_service.dart';
import 'package:pest_assessment/services/history_store.dart';
import 'package:pest_assessment/services/scan_service.dart';
import 'package:pest_assessment/strings.dart';
import 'package:shared_preferences/shared_preferences.dart';

class FakeAuth implements AuthService {
  FakeAuth({this.confirmEmail = false});

  /// Starts signed in as Juan from La Union.
  FakeAuth.signedIn() : confirmEmail = false {
    users['juan@example.com'] = (password: 'secret123', name: 'Juan Dela Cruz', barangay: 13);
    current = 'juan@example.com';
  }

  final bool confirmEmail; // like Supabase's "Confirm email" setting
  final _changes = StreamController<bool>.broadcast();
  final users = <String, ({String password, String name, int barangay})>{};
  String? current;
  final unconfirmed = <String>{};

  @override
  bool get isSignedIn => current != null;
  @override
  Stream<bool> get signedInChanges => _changes.stream;

  @override
  Future<void> signIn({required String email, required String password}) async {
    final u = users[email.trim()];
    if (u == null || u.password != password) throw const AuthFailure(S.wrongLogin);
    if (unconfirmed.contains(email.trim())) throw const AuthFailure(S.emailNotConfirmed);
    current = email.trim();
    _changes.add(true);
  }

  @override
  Future<bool> signUp({required String fullName, required int barangayId, required String email, required String password}) async {
    if (users.containsKey(email)) throw const AuthFailure(S.emailTaken);
    users[email] = (password: password, name: fullName, barangay: barangayId);
    if (confirmEmail) {
      unconfirmed.add(email);
      codesSent.add(email); // the confirmation email has a code
      return false;
    }
    current = email;
    _changes.add(true);
    return true;
  }

  @override
  Future<void> signOut() async {
    current = null;
    _changes.add(false);
  }

  /// The code every fake email contains.
  static const goodCode = '123456';
  final codesSent = <String>[];

  @override
  Future<void> sendCode(String email) async => codesSent.add(email.trim());

  @override
  Future<void> verifyCode({required String email, required String code}) async {
    if (!codesSent.contains(email.trim()) || code.trim() != goodCode) throw const AuthFailure(S.codeWrong);
    users.putIfAbsent(email.trim(), () => (password: '', name: '', barangay: 0)); // new account
    unconfirmed.remove(email.trim());
    current = email.trim();
    _changes.add(true);
  }

  @override
  Future<void> changePassword(String newPassword) async {
    final u = users[current]!;
    users[current!] = (password: newPassword, name: u.name, barangay: u.barangay);
  }

  @override
  Future<void> updateProfile({required String fullName, required int barangayId}) async {
    final u = users[current]!;
    users[current!] = (password: u.password, name: fullName.trim(), barangay: barangayId);
  }

  @override
  Future<List<Barangay>> barangays() async => const [Barangay(1, 'Antonio Luna'), Barangay(13, 'La Union')];

  @override
  Future<Profile> profile() async {
    final u = users[current];
    if (u == null || u.barangay == 0) return Profile(fullName: u?.name, email: current);
    return Profile(
        fullName: u.name, barangay: u.barangay == 13 ? 'La Union' : 'Antonio Luna', barangayId: u.barangay, email: current);
  }
}

/// Answers in order (the last one repeats). A ScanFailure in the list is thrown.
class FakeApi implements ScanApi {
  FakeApi(this.answers);
  final List<Object> answers;
  int calls = 0;

  @override
  Future<ScanResult> check(Uint8List photo) async {
    final a = answers[min(calls, answers.length - 1)];
    calls++;
    if (a is ScanFailure) throw a;
    return a as ScanResult;
  }
}

/// Supabase stand-in for History; set [online] to false to test working offline.
class FakeRemote implements ScanRemote {
  bool online = true;
  @override
  String? userId = 'user-1';
  final rows = <String, LocalScan>{};
  final removed = <String>[];
  final fromOtherPhone = <RemoteScan>[];

  void _check() {
    if (!online) throw Exception('no internet');
  }

  @override
  Future<void> upload(LocalScan scan, Uint8List photo) async {
    _check();
    rows[scan.id] = scan;
  }

  @override
  Future<void> remove(LocalScan scan) async {
    _check();
    rows.remove(scan.id);
    removed.add(scan.id);
  }

  @override
  Future<List<RemoteScan>> list() async {
    _check();
    return [
      ...fromOtherPhone,
      for (final s in rows.values)
        RemoteScan(id: s.id, createdAt: s.createdAt, result: s.result, photoPath: s.remotePhotoPath),
    ];
  }

  @override
  Future<Uint8List> downloadPhoto(String path) async {
    _check();
    return tinyPng;
  }
}

/// Phone storage that fails the next [failures] writes (e.g. storage full).
class FlakyStorage extends MemoryStorage {
  int failures = 0;
  @override
  Future<void> write(String name, List<int> bytes) async {
    if (failures > 0) {
      failures--;
      throw Exception('storage full');
    }
    return super.write(name, bytes);
  }
}

/// A real 1x1 PNG, so Image.memory can show it.
final tinyPng = base64Decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==');

class FakePhotos implements PhotoSource {
  Uint8List? next = tinyPng; // null = the user cancelled
  bool? lastCamera;

  @override
  Future<Uint8List?> pick({required bool camera}) async {
    lastCamera = camera;
    return next;
  }
}

Future<AppServices> makeServices({
  FakeAuth? auth,
  ScanApi? api,
  HistoryStore? history,
  PhotoSource? photos,
  Map<String, Object> prefs = const {},
}) async {
  SharedPreferences.setMockInitialValues(prefs);
  return AppServices(
    auth: auth ?? FakeAuth(),
    api: api ?? FakeApi(const [ScanResult(outcome: 'healthy', severity: 'Healthy')]),
    history: history ?? HistoryStore(storage: MemoryStorage(), remote: FakeRemote()),
    photos: photos ?? FakePhotos(),
    tips: TipsPrefs(await SharedPreferences.getInstance()),
  );
}

/// Opens the app on the email + password login form (the first screen).
Future<void> pumpPasswordApp(WidgetTester tester, AppServices services) => pumpApp(tester, services);

/// Opens the app and goes to "Sign in with an email code instead".
Future<void> pumpCodeApp(WidgetTester tester, AppServices services) async {
  await pumpApp(tester, services);
  await tester.tap(find.text(S.useCode));
  await tester.pumpAndSettle();
}

/// Signs out from the Settings tab.
Future<void> signOutViaSettings(WidgetTester tester) async {
  await tester.tap(find.byTooltip(S.settings));
  await tester.pumpAndSettle();
  await tester.tap(find.widgetWithText(OutlinedButton, S.signOut));
  await tester.pumpAndSettle();
  await tester.tap(find.widgetWithText(FilledButton, S.signOut));
  await tester.pumpAndSettle();
}

Future<void> pumpApp(WidgetTester tester, AppServices services) async {
  tester.view.physicalSize = const Size(1080, 2400);
  tester.view.devicePixelRatio = 2.5;
  addTearDown(tester.view.reset);
  await tester.pumpWidget(CornPestApp(services: services));
  await tester.pumpAndSettle();
}
