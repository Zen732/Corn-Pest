import 'dart:typed_data';

import 'package:flutter/material.dart';

import '../services/app_services.dart';
import '../services/auth_service.dart';
import '../services/history_store.dart';
import '../strings.dart';
import 'history_screen.dart';

/// The Home tab: greeting, the user's numbers, recent scans and a photo tip.
class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key, required this.services, this.onSeeAll});
  final AppServices services;

  /// Opens the History tab.
  final VoidCallback? onSeeAll;

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  late Future<Profile> _profile = widget.services.auth.profile();
  late Future<List<LocalScan>> _scans = widget.services.history.list();

  @override
  void initState() {
    super.initState();
    widget.services.history.changes.addListener(_reload);
  }

  @override
  void dispose() {
    widget.services.history.changes.removeListener(_reload);
    super.dispose();
  }

  void _reload() {
    if (!mounted) return;
    setState(() {
      _scans = widget.services.history.list();
    });
  }

  /// Called by Settings after the name or barangay changes.
  void reloadProfile() => setState(() {
        _profile = widget.services.auth.profile();
      });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final scheme = theme.colorScheme;
    return Scaffold(
      body: RefreshIndicator(
        onRefresh: () async {
          await widget.services.history.sync();
          reloadProfile();
        },
        child: ListView(
          padding: EdgeInsets.zero,
          children: [
            // Header
            Container(
              padding: EdgeInsets.fromLTRB(24, MediaQuery.paddingOf(context).top + 24, 24, 28),
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: [scheme.primary, const Color(0xFF66BB6A)],
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                ),
                borderRadius: const BorderRadius.vertical(bottom: Radius.circular(28)),
              ),
              child: FutureBuilder<Profile>(
                future: _profile,
                builder: (context, snap) {
                  final p = snap.data;
                  const white = TextStyle(color: Colors.white);
                  return Row(children: [
                    Expanded(
                      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                        Text(S.today(), style: white.copyWith(fontSize: 14, color: Colors.white70)),
                        const SizedBox(height: 4),
                        Text(S.hello(p?.fullName), style: theme.textTheme.headlineSmall?.copyWith(color: Colors.white)),
                        if (p?.barangay != null) ...[
                          const SizedBox(height: 4),
                          Row(children: [
                            const Icon(Icons.place, size: 16, color: Colors.white70),
                            const SizedBox(width: 4),
                            Flexible(
                              child: Text(S.fromBarangay(p!.barangay!),
                                  overflow: TextOverflow.ellipsis, style: white.copyWith(fontSize: 15)),
                            ),
                          ]),
                        ],
                      ]),
                    ),
                    const Icon(Icons.eco, size: 56, color: Colors.white24),
                  ]);
                },
              ),
            ),
            FutureBuilder<List<LocalScan>>(
              future: _scans,
              builder: (context, snap) {
                final scans = snap.data ?? const <LocalScan>[];
                final pests = scans.where((s) => s.result.outcome == 'pest').length;
                final waiting = scans.where((s) => !s.synced).length;
                return Padding(
                  padding: const EdgeInsets.fromLTRB(16, 16, 16, 0),
                  child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
                    Row(children: [
                      _Stat(icon: Icons.photo_camera_outlined, value: scans.length, label: S.statScans),
                      _Stat(icon: Icons.bug_report_outlined, value: pests, label: S.statPests, color: const Color(0xFFC62828)),
                      _Stat(icon: Icons.cloud_upload_outlined, value: waiting, label: S.statWaiting, color: Colors.grey.shade700),
                    ]),
                    const SizedBox(height: 20),
                    Row(children: [
                      Expanded(child: Text(S.recentScans, style: theme.textTheme.titleMedium)),
                      if (scans.isNotEmpty) TextButton(onPressed: widget.onSeeAll, child: const Text(S.seeAll)),
                    ]),
                    if (snap.connectionState != ConnectionState.done)
                      const Padding(padding: EdgeInsets.all(16), child: Center(child: CircularProgressIndicator()))
                    else if (scans.isEmpty)
                      Card(
                        child: Padding(
                          padding: const EdgeInsets.all(20),
                          child: Row(children: [
                            Icon(Icons.photo_camera, color: scheme.primary, size: 32),
                            const SizedBox(width: 16),
                            const Expanded(child: Text(S.noScansYet)),
                          ]),
                        ),
                      )
                    else
                      Card(
                        clipBehavior: Clip.antiAlias,
                        child: Column(children: [
                          for (final s in scans.take(3))
                            ListTile(
                              leading: _Thumb(history: widget.services.history, scan: s),
                              title: Text(S.summary(s.result.outcome, s.result.pest, s.result.severity)),
                              subtitle: Text(S.scannedOn(s.createdAt)),
                              trailing: const Icon(Icons.chevron_right),
                              onTap: () => Navigator.of(context).push(MaterialPageRoute(
                                builder: (_) => ScanDetailScreen(services: widget.services, scan: s),
                              )),
                            ),
                        ]),
                      ),
                  ]),
                );
              },
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 12, 16, 24),
              child: Card(
                color: scheme.secondaryContainer,
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
                    Icon(Icons.lightbulb_outline, color: scheme.onSecondaryContainer),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                        Text(S.tipTitle, style: theme.textTheme.titleSmall),
                        const SizedBox(height: 4),
                        Text(S.tips[DateTime.now().day % S.tips.length]),
                      ]),
                    ),
                  ]),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _Stat extends StatelessWidget {
  const _Stat({required this.icon, required this.value, required this.label, this.color});
  final IconData icon;
  final int value;
  final String label;
  final Color? color;

  @override
  Widget build(BuildContext context) {
    final c = color ?? Theme.of(context).colorScheme.primary;
    return Expanded(
      child: Card(
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 8),
          child: Column(children: [
            Icon(icon, color: c),
            const SizedBox(height: 4),
            Text('$value', style: Theme.of(context).textTheme.headlineSmall?.copyWith(color: c)),
            Text(label, textAlign: TextAlign.center, style: Theme.of(context).textTheme.bodySmall),
          ]),
        ),
      ),
    );
  }
}

class _Thumb extends StatefulWidget {
  const _Thumb({required this.history, required this.scan});
  final HistoryStore history;
  final LocalScan scan;

  @override
  State<_Thumb> createState() => _ThumbState();
}

class _ThumbState extends State<_Thumb> {
  late final Future<Uint8List?> _photo = widget.history.photo(widget.scan);

  @override
  Widget build(BuildContext context) => SizedBox(
        width: 48,
        height: 48,
        child: ClipRRect(
          borderRadius: BorderRadius.circular(8),
          child: FutureBuilder<Uint8List?>(
            future: _photo,
            builder: (context, snap) => snap.data == null
                ? const ColoredBox(color: Color(0x11000000), child: Icon(Icons.image_outlined))
                : Image.memory(snap.data!, fit: BoxFit.cover, cacheWidth: 144),
          ),
        ),
      );
}
