import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

import 'config.dart';
import 'screens/home_screen.dart';
import 'screens/sign_in_screen.dart';
import 'services/app_services.dart';
import 'services/auth_service.dart';
import 'services/scan_service.dart';
import 'strings.dart';

/// Corn Pest Assessment (Cabadbaran). Run with:
///     flutter run --dart-define-from-file=env.json
Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  if (!Config.hasSupabase) {
    runApp(const SetupMissingApp());
    return;
  }
  await Supabase.initialize(url: Config.supabaseUrl, publishableKey: Config.supabaseKey.trim());
  runApp(CornPestApp(
    services: AppServices(
      auth: SupabaseAuthService(),
      api: BackendScanApi(),
      store: SupabaseScanStore(),
      photos: ImagePickerSource(),
      tips: TipsPrefs(await SharedPreferences.getInstance()),
    ),
  ));
}

ThemeData buildTheme() {
  final scheme = ColorScheme.fromSeed(seedColor: const Color(0xFF2E7D32)); // corn-leaf green
  return ThemeData(
    colorScheme: scheme,
    useMaterial3: true,
    inputDecorationTheme: const InputDecorationTheme(border: OutlineInputBorder()),
    filledButtonTheme: FilledButtonThemeData(
      style: FilledButton.styleFrom(minimumSize: const Size.fromHeight(56), textStyle: const TextStyle(fontSize: 18)),
    ),
    outlinedButtonTheme: OutlinedButtonThemeData(
      style: OutlinedButton.styleFrom(minimumSize: const Size.fromHeight(56), textStyle: const TextStyle(fontSize: 18)),
    ),
  );
}

class CornPestApp extends StatelessWidget {
  const CornPestApp({super.key, required this.services});
  final AppServices services;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: S.appName,
      theme: buildTheme(),
      debugShowCheckedModeBanner: false,
      home: AuthGate(services: services),
    );
  }
}

/// Shows Sign in or Home, and switches by itself when the user signs in or out.
class AuthGate extends StatelessWidget {
  const AuthGate({super.key, required this.services});
  final AppServices services;

  AuthService get auth => services.auth;

  @override
  Widget build(BuildContext context) {
    return StreamBuilder<bool>(
      stream: auth.signedInChanges,
      initialData: auth.isSignedIn,
      builder: (context, snap) => (snap.data ?? false)
          ? HomeScreen(key: const ValueKey('home'), services: services)
          : SignInScreen(auth: auth),
    );
  }
}

class SetupMissingApp extends StatelessWidget {
  const SetupMissingApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      theme: buildTheme(),
      home: const Scaffold(
        body: Center(
          child: Padding(
            padding: EdgeInsets.all(24),
            child: Column(mainAxisSize: MainAxisSize.min, children: [
              Icon(Icons.settings, size: 64),
              SizedBox(height: 16),
              Text(S.setupMissingTitle, style: TextStyle(fontSize: 22)),
              SizedBox(height: 12),
              Text(S.setupMissingBody, textAlign: TextAlign.center),
            ]),
          ),
        ),
      ),
    );
  }
}
