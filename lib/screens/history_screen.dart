import 'dart:typed_data';

import 'package:flutter/material.dart';

import '../services/app_services.dart';
import '../services/history_store.dart';
import '../strings.dart';
import '../widgets/result_card.dart';

/// The user's scans, newest first. Works offline; pull down to upload and refresh.
class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key, required this.services});
  final AppServices services;

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  HistoryStore get _history => widget.services.history;
  late Future<List<LocalScan>> _scans = _history.list();

  @override
  void initState() {
    super.initState();
    _history.changes.addListener(_reload);
    _history.sync(); // upload anything waiting, and pick up scans from other phones
  }

  @override
  void dispose() {
    _history.changes.removeListener(_reload);
    super.dispose();
  }

  void _reload() {
    if (!mounted) return;
    setState(() {
      _scans = _history.list();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text(S.history)),
      body: RefreshIndicator(
        onRefresh: () async {
          await _history.sync();
          _reload();
        },
        child: FutureBuilder<List<LocalScan>>(
          future: _scans,
          builder: (context, snap) {
            final scans = snap.data;
            if (scans == null) return const Center(child: CircularProgressIndicator());
            if (scans.isEmpty) {
              return ListView(children: const [
                Padding(padding: EdgeInsets.all(32), child: Text(S.historyEmpty, textAlign: TextAlign.center)),
              ]);
            }
            final waiting = scans.where((s) => !s.synced).length;
            return ListView(
              children: [
                if (waiting > 0)
                  ListTile(
                    leading: const Icon(Icons.cloud_upload_outlined),
                    title: Text(S.waitingCount(waiting)),
                    subtitle: const Text(S.waitingNote),
                  ),
                for (final s in scans)
                  ListTile(
                    leading: _Thumb(history: _history, scan: s),
                    title: Text(S.summary(s.result.outcome, s.result.pest, s.result.severity)),
                    subtitle: Text(S.scannedOn(s.createdAt)),
                    trailing: Icon(
                      s.synced ? Icons.cloud_done : Icons.cloud_upload_outlined,
                      semanticLabel: s.synced ? S.uploaded : S.waitingToUpload,
                      color: s.synced ? const Color(0xFF2E7D32) : Colors.grey,
                    ),
                    onTap: () => Navigator.of(context).push(MaterialPageRoute(
                      builder: (_) => ScanDetailScreen(services: widget.services, scan: s),
                    )),
                  ),
              ],
            );
          },
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
        width: 56,
        height: 56,
        child: ClipRRect(
          borderRadius: BorderRadius.circular(8),
          child: FutureBuilder<Uint8List?>(
            future: _photo,
            builder: (context, snap) => snap.data == null
                ? const ColoredBox(color: Color(0x11000000), child: Icon(Icons.image_outlined))
                : Image.memory(snap.data!, fit: BoxFit.cover, cacheWidth: 168),
          ),
        ),
      );
}

/// One scan: photo, result, date, upload state, and Delete.
class ScanDetailScreen extends StatelessWidget {
  const ScanDetailScreen({super.key, required this.services, required this.scan});
  final AppServices services;
  final LocalScan scan;

  Future<void> _delete(BuildContext context) async {
    final yes = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text(S.deleteQuestion),
        actions: [
          TextButton(onPressed: () => Navigator.of(context).pop(false), child: const Text(S.cancel)),
          FilledButton(onPressed: () => Navigator.of(context).pop(true), child: const Text(S.delete)),
        ],
      ),
    );
    if (yes != true) return;
    await services.history.delete(scan);
    if (context.mounted) Navigator.of(context).pop();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text(S.scanDetails)),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(24),
          children: [
            ClipRRect(
              borderRadius: BorderRadius.circular(16),
              child: FutureBuilder<Uint8List?>(
                future: services.history.photo(scan),
                builder: (context, snap) => snap.data == null
                    ? SizedBox(
                        height: 120,
                        child: Center(
                          child: snap.connectionState == ConnectionState.done
                              ? const Text(S.photoNotAvailable, textAlign: TextAlign.center)
                              : const CircularProgressIndicator(),
                        ),
                      )
                    : Image.memory(snap.data!, height: 240, fit: BoxFit.cover),
              ),
            ),
            const SizedBox(height: 16),
            Text(S.scannedOn(scan.createdAt), textAlign: TextAlign.center),
            const SizedBox(height: 16),
            ResultCard(result: scan.result),
            const SizedBox(height: 12),
            Row(mainAxisAlignment: MainAxisAlignment.center, children: [
              Icon(scan.synced ? Icons.cloud_done : Icons.cloud_upload_outlined,
                  color: scan.synced ? const Color(0xFF2E7D32) : Colors.grey),
              const SizedBox(width: 8),
              Flexible(child: Text(scan.synced ? S.uploaded : S.waitingNote)),
            ]),
            const SizedBox(height: 24),
            OutlinedButton.icon(
              onPressed: () => _delete(context),
              icon: const Icon(Icons.delete_outline),
              label: const Text(S.deleteScan),
            ),
          ],
        ),
      ),
    );
  }
}
