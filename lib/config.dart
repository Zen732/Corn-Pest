/// Settings read when the app is built, from env.json (which stays on your PC):
///
///     flutter run --dart-define-from-file=env.json
///
/// See env.example.json for the keys.
class Config {
  static const _supabaseUrl = String.fromEnvironment('SUPABASE_URL');
  static const supabaseKey = String.fromEnvironment('SUPABASE_ANON_KEY');
  static const backendUrl = String.fromEnvironment('BACKEND_URL');

  static String get supabaseUrl => normalizeSupabaseUrl(_supabaseUrl);

  static bool get hasSupabase => supabaseUrl.startsWith('https://') && supabaseKey.trim().isNotEmpty;

  /// "https://abc.supabase.co", also when the REST address ".../rest/v1/" was pasted.
  static String normalizeSupabaseUrl(String url) =>
      url.trim().replaceFirst(RegExp(r'(/rest/v1)?/*$'), '');
}
