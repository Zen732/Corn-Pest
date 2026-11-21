import 'dart:typed_data';

import 'package:flutter/material.dart';

import '../services/app_services.dart';
import '../services/scan_service.dart';
import '../strings.dart';
import '../widgets/result_card.dart';

enum _Save { none, saving, saved, failed }

/// Sends the photo to the backend, shows the result, and saves real results to History.
/// No treatment advice is shown anywhere.
class ResultScreen extends StatefulWidget {
  const ResultScreen({super.key, required this.services, required this.photo});
  final AppServices services;
  final Uint8List photo;

  @override
  State<ResultScreen> createState() => _ResultScreenState();
}

class _ResultScreenState extends State<ResultScreen> {
  ScanResult? _result;
  String? _error;
  _Save _save = _Save.none;

  @override
  void initState() {
    super.initState();
    _check();
  }

  Future<void> _check() async {
    setState(() {
      _result = null;
      _error = null;
    });
    try {
      final result = await widget.services.api.check(widget.photo);
      if (!mounted) return;
      setState(() => _result = result);
      if (result.savable) await _saveResult();
    } on ScanFailure catch (e) {
      if (mounted) setState(() => _error = e.message);
    }
  }

  Future<void> _saveResult() async {
    setState(() => _save = _Save.saving);
    try {
      // Saved on the phone (works offline); it uploads to Supabase by itself when online.
      await widget.services.history.save(_result!, widget.photo);
      if (mounted) setState(() => _save = _Save.saved);
    } catch (_) {
      if (mounted) setState(() => _save = _Save.failed); // e.g. the phone's storage is full
    }
  }

  @override
  Widget build(BuildContext context) {
    final result = _result;
    final retake = result != null && !result.savable;
    return Scaffold(
      appBar: AppBar(title: const Text(S.scanPlant)),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(24),
          children: [
            ClipRRect(
              borderRadius: BorderRadius.circular(16),
              child: Image.memory(widget.photo, height: 240, fit: BoxFit.cover, gaplessPlayback: true),
            ),
            const SizedBox(height: 24),
            if (_error != null)
              MessageCard(icon: Icons.cloud_off, color: Colors.grey.shade700, title: _error!)
            else if (result == null)
              const Column(children: [
                CircularProgressIndicator(),
                SizedBox(height: 16),
                Text(S.checking, style: TextStyle(fontSize: 18)),
              ])
            else
              ResultCard(result: result),
            if (result != null && result.savable) ...[
              const SizedBox(height: 16),
              _SaveStatus(state: _save, onRetry: _saveResult),
            ],
            const SizedBox(height: 24),
            if (_error != null) ...[
              FilledButton(onPressed: _check, child: const Text(S.tryAgain)),
              const SizedBox(height: 12),
              OutlinedButton(onPressed: () => Navigator.of(context).pop(), child: const Text(S.back)),
            ] else if (result != null) ...[
              FilledButton.icon(
                onPressed: () => Navigator.of(context).pop(),
                icon: const Icon(Icons.photo_camera),
                label: Text(retake ? S.retakePhoto : S.scanAnother),
              ),
              const SizedBox(height: 12),
              OutlinedButton(
                onPressed: _save == _Save.saving ? null : () => Navigator.of(context).popUntil((r) => r.isFirst),
                child: const Text(S.done),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class _SaveStatus extends StatelessWidget {
  const _SaveStatus({required this.state, required this.onRetry});
  final _Save state;
  final VoidCallback onRetry;

  @override
  Widget build(BuildContext context) {
    switch (state) {
      case _Save.saving:
        return const Row(mainAxisAlignment: MainAxisAlignment.center, children: [
          SizedBox(width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2)),
          SizedBox(width: 12),
          Text(S.saving),
        ]);
      case _Save.saved:
        return const Row(mainAxisAlignment: MainAxisAlignment.center, children: [
          Icon(Icons.cloud_done, color: Color(0xFF2E7D32)),
          SizedBox(width: 8),
          Text(S.saved),
        ]);
      case _Save.failed:
        return Column(children: [
          const Text(S.saveFailed, textAlign: TextAlign.center),
          TextButton(onPressed: onRetry, child: const Text(S.tryAgain)),
        ]);
      case _Save.none:
        return const SizedBox.shrink();
    }
  }
}
