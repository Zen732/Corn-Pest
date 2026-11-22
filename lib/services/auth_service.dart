import 'package:supabase_flutter/supabase_flutter.dart';

import '../strings.dart';

class Barangay {
  const Barangay(this.id, this.name);
  final int id;
  final String name;
}

class Profile {
  const Profile({this.fullName, this.barangay, this.barangayId, this.email});
  final String? fullName;
  final String? barangay;
  final int? barangayId;
  final String? email;

  /// New accounts made with an email code have no name or barangay yet.
  bool get complete => barangay != null && (fullName ?? '').trim().isNotEmpty;
}

/// A problem to show the user, already in plain words.
class AuthFailure implements Exception {
  const AuthFailure(this.message);
  final String message;
  @override
  String toString() => message;
}

/// Login and the user's profile. The screens only talk to this, so the tests can use a fake.
abstract class AuthService {
  bool get isSignedIn;

  /// Emits true when the user signs in and false when they sign out.
  Stream<bool> get signedInChanges;

  Future<void> signIn({required String email, required String password});

  /// Emails a sign-in code (and makes the account if it's new).
  Future<void> sendCode(String email);

  /// Signs in with the code from the email.
  Future<void> verifyCode({required String email, required String code});

  /// Saves the name and barangay (asked once, after the first sign-in with a code).
  Future<void> updateProfile({required String fullName, required int barangayId});

  /// Sets a new password (also lets someone who signed in with a code use a password).
  Future<void> changePassword(String newPassword);

  /// Returns true if the user is signed in right away, false if they must confirm their
  /// email first (when "Confirm email" is on in Supabase): Supabase then emails a 6-digit
  /// code, which [verifyCode] checks.
  Future<bool> signUp({
    required String fullName,
    required int barangayId,
    required String email,
    required String password,
  });

  Future<void> signOut();

  Future<List<Barangay>> barangays();

  Future<Profile> profile();
}

class SupabaseAuthService implements AuthService {
  SupabaseClient get _db => Supabase.instance.client;

  @override
  bool get isSignedIn => _db.auth.currentSession != null;

  @override
  Stream<bool> get signedInChanges => _db.auth.onAuthStateChange.map((s) => s.session != null);

  @override
  Future<void> signIn({required String email, required String password}) =>
      _guard(() => _db.auth.signInWithPassword(email: email.trim(), password: password));

  @override
  Future<void> sendCode(String email) =>
      _guard(() => _db.auth.signInWithOtp(email: email.trim(), shouldCreateUser: true));

  @override
  Future<void> verifyCode({required String email, required String code}) =>
      _guard(() => _db.auth.verifyOTP(email: email.trim(), token: code.trim(), type: OtpType.email));

  @override
  Future<void> updateProfile({required String fullName, required int barangayId}) => _guard(() async {
        final uid = _db.auth.currentUser?.id;
        if (uid == null) throw const AuthFailure(S.couldNotConnect);
        await _db.from('profiles').update({'full_name': fullName.trim(), 'barangay_id': barangayId}).eq('id', uid);
      });

  @override
  Future<void> changePassword(String newPassword) =>
      _guard(() => _db.auth.updateUser(UserAttributes(password: newPassword)));

  @override
  Future<bool> signUp({
    required String fullName,
    required int barangayId,
    required String email,
    required String password,
  }) =>
      _guard(() async {
        // The database copies full_name and barangay_id into the user's profile
        // (database/schema.sql, handle_new_user).
        final res = await _db.auth.signUp(
          email: email.trim(),
          password: password,
          data: {'full_name': fullName.trim(), 'barangay_id': '$barangayId'},
        );
        if (res.user != null && (res.user!.identities ?? const []).isEmpty) {
          throw const AuthFailure(S.emailTaken); // Supabase's answer for an existing email
        }
        return res.session != null;
      });

  @override
  Future<void> signOut() => _guard(() => _db.auth.signOut());

  @override
  Future<List<Barangay>> barangays() => _guard(() async {
        final rows = await _db.from('barangays').select('id, name').order('id');
        return [for (final r in rows) Barangay(r['id'] as int, r['name'] as String)];
      });

  @override
  Future<Profile> profile() => _guard(() async {
        final uid = _db.auth.currentUser?.id;
        if (uid == null) return const Profile();
        final row =
            await _db.from('profiles').select('full_name, barangay_id, barangays(name)').eq('id', uid).maybeSingle();
        return Profile(
          fullName: row?['full_name'] as String?,
          barangay: (row?['barangays'] as Map<String, dynamic>?)?['name'] as String?,
          barangayId: row?['barangay_id'] as int?,
          email: _db.auth.currentUser?.email,
        );
      });

  /// Turns Supabase and network errors into messages a farmer can act on.
  Future<T> _guard<T>(Future<T> Function() action) async {
    try {
      return await action();
    } on AuthFailure {
      rethrow;
    } on AuthException catch (e) {
      final m = e.message.toLowerCase();
      if (m.contains('invalid login')) throw const AuthFailure(S.wrongLogin);
      if (m.contains('not confirmed')) throw const AuthFailure(S.emailNotConfirmed);
      if (m.contains('already registered') || m.contains('already been registered')) {
        throw const AuthFailure(S.emailTaken);
      }
      if (m.contains('password') && m.contains('characters')) throw const AuthFailure(S.passwordTooShort);
      if (m.contains('expired') || m.contains('invalid') || m.contains('otp')) throw const AuthFailure(S.codeWrong);
      if (m.contains('rate limit') || m.contains('security purposes') || m.contains('seconds')) {
        throw const AuthFailure(S.codeTooSoon);
      }
      if (m.contains('sending') && m.contains('email')) throw const AuthFailure(S.codeSendFailed);
      if (e is AuthRetryableFetchException) throw const AuthFailure(S.couldNotConnect);
      throw AuthFailure(e.message);
    } on PostgrestException catch (e) {
      throw AuthFailure(e.message);
    } catch (_) {
      throw const AuthFailure(S.couldNotConnect);
    }
  }
}
