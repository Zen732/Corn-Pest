import 'package:flutter/material.dart';

import '../services/app_services.dart';
import '../services/auth_service.dart';
import '../strings.dart';
import 'scan_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key, required this.services});
  final AppServices services;

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  late Future<Profile> _profile;

  @override
  void initState() {
    super.initState();
    _profile = widget.services.auth.profile();
  }

  Future<void> _signOut() async {
    final yes = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text(S.signOutQuestion),
        actions: [
          TextButton(onPressed: () => Navigator.of(context).pop(false), child: const Text(S.cancel)),
          FilledButton(onPressed: () => Navigator.of(context).pop(true), child: const Text(S.signOut)),
        ],
      ),
    );
    if (yes == true) {
      try {
        await widget.services.auth.signOut();
      } on AuthFailure catch (e) {
        if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(e.message)));
      }
    }
  }

  void _comingSoon() =>
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text(S.comingSoon)));

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(
        title: const Text(S.appName),
        actions: [IconButton(tooltip: S.signOut, icon: const Icon(Icons.logout), onPressed: _signOut)],
      ),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(24),
          children: [
            FutureBuilder<Profile>(
              future: _profile,
              builder: (context, snap) {
                final p = snap.data;
                return Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(S.hello(p?.fullName), style: theme.textTheme.headlineSmall),
                    if (p?.barangay != null) ...[
                      const SizedBox(height: 4),
                      Text(S.fromBarangay(p!.barangay!), style: theme.textTheme.bodyLarge),
                    ],
                  ],
                );
              },
            ),
            const SizedBox(height: 32),
            SizedBox(
              height: 120,
              child: FilledButton.icon(
                onPressed: () => Navigator.of(context)
                    .push(MaterialPageRoute(builder: (_) => ScanScreen(services: widget.services))),
                icon: const Icon(Icons.photo_camera, size: 36),
                label: const Text(S.scanPlant, style: TextStyle(fontSize: 22)),
              ),
            ),
            const SizedBox(height: 16),
            SizedBox(
              height: 72,
              child: OutlinedButton.icon(
                onPressed: _comingSoon,
                icon: const Icon(Icons.history, size: 28),
                label: const Text(S.history, style: TextStyle(fontSize: 20)),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
