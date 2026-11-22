import 'dart:async';

import 'package:flutter/material.dart';

import '../services/app_services.dart';
import '../services/scan_service.dart';
import '../strings.dart';
import 'history_screen.dart';
import 'home_screen.dart';
import 'scan_screen.dart';
import 'settings_screen.dart';

/// The signed-in app: Home, History and Settings tabs in a bottom bar (icons only), with a big
/// round camera button in the middle to scan a plant.
class MainShell extends StatefulWidget {
  const MainShell({super.key, required this.services});
  final AppServices services;

  @override
  State<MainShell> createState() => _MainShellState();
}

class _MainShellState extends State<MainShell> {
  int _tab = 0;
  Key _homeKey = UniqueKey();
  Timer? _syncTimer;
  late final Future<String?>? _modelVersion = widget.services.api is OfflineScanApi
      ? (widget.services.api as OfflineScanApi).load().then<String?>((m) => m.version).catchError((_) => null)
      : null;

  @override
  void initState() {
    super.initState();
    // Upload scans saved while offline: now, and every minute while the app is open.
    widget.services.history.sync();
    _syncTimer = Timer.periodic(const Duration(minutes: 1), (_) => widget.services.history.sync());
  }

  @override
  void dispose() {
    _syncTimer?.cancel();
    super.dispose();
  }

  void _scan() => Navigator.of(context).push(MaterialPageRoute(builder: (_) => ScanScreen(services: widget.services)));

  Widget _tabButton(int index, IconData icon, IconData selectedIcon, String label) {
    final selected = _tab == index;
    final color = Theme.of(context).colorScheme.primary;
    return IconButton(
      key: Key('tab-$index'),
      tooltip: label, // read aloud by screen readers; the bar itself shows only icons
      iconSize: 30,
      isSelected: selected,
      icon: Icon(icon, color: selected ? color : Colors.grey.shade600),
      selectedIcon: Icon(selectedIcon, color: color),
      onPressed: () => setState(() => _tab = index),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(
        index: _tab,
        children: [
          HomeScreen(key: _homeKey, services: widget.services, onSeeAll: () => setState(() => _tab = 1)),
          HistoryScreen(services: widget.services),
          SettingsScreen(
            services: widget.services,
            modelVersion: _modelVersion,
            onProfileChanged: () => setState(() => _homeKey = UniqueKey()),
          ),
        ],
      ),
      floatingActionButton: FloatingActionButton.large(
        key: const Key('scan-button'),
        tooltip: S.scanPlant,
        shape: const CircleBorder(),
        onPressed: _scan,
        child: const Icon(Icons.photo_camera, size: 40),
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerDocked,
      bottomNavigationBar: BottomAppBar(
        shape: const CircularNotchedRectangle(),
        notchMargin: 8,
        height: 72,
        child: Row(children: [
          Expanded(
            child: Row(mainAxisAlignment: MainAxisAlignment.spaceEvenly, children: [
              _tabButton(0, Icons.home_outlined, Icons.home, S.home),
              _tabButton(1, Icons.history, Icons.history, S.history),
            ]),
          ),
          const SizedBox(width: 96), // room for the camera button
          Expanded(
            child: Row(mainAxisAlignment: MainAxisAlignment.spaceEvenly, children: [
              _tabButton(2, Icons.settings_outlined, Icons.settings, S.settings),
            ]),
          ),
        ]),
      ),
    );
  }
}
