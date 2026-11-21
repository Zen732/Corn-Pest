import 'auth_service.dart';
import 'history_store.dart';
import 'scan_service.dart';

/// Everything the screens talk to, in one bundle. The app uses the real ones (main.dart);
/// the tests pass fakes, so they need no internet, camera or Supabase.
class AppServices {
  const AppServices({required this.auth, required this.api, required this.history, required this.photos, required this.tips});
  final AuthService auth;
  final ScanApi api;
  final HistoryStore history;
  final PhotoSource photos;
  final TipsPrefs tips;
}
