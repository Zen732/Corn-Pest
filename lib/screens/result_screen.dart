import 'dart:typed_data';

import 'package:flutter/material.dart';

import '../services/app_services.dart';
import '../services/scan_service.dart';
import '../strings.dart';

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
      await widget.services.store.save(_result!, widget.photo);
      if (mounted) setState(() => _save = _Save.saved);
    } on ScanFailure {
      if (mounted) setState(() => _save = _Save.failed);
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
              _Message(icon: Icons.cloud_off, color: Colors.grey.shade700, title: _error!)
            else if (result == null)
              const Column(children: [
                CircularProgressIndicator(),
                SizedBox(height: 16),
                Text(S.checking, style: TextStyle(fontSize: 18)),
              ])
            else
              _ResultCard(result: result),
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

class _ResultCard extends StatelessWidget {
  const _ResultCard({required this.result});
  final ScanResult result;

  @override
  Widget build(BuildContext context) {
    final r = result;
    switch (r.outcome) {
      case 'healthy':
        return const _Message(icon: Icons.check_circle, color: Color(0xFF2E7D32), title: S.healthyTitle, lines: [S.healthyBody]);
      case 'severity_only':
        return _Message(
          icon: Icons.warning_amber_rounded,
          color: const Color(0xFFE65100),
          title: S.damageTitle,
          lines: [S.severityLine(r.severity ?? '', r.damagePercent), S.noInsectSeen],
        );
      case 'pest':
        final (name, scientific) = S.pests[r.pest] ?? (r.pest ?? '', '');
        return _Message(
          icon: Icons.bug_report,
          color: const Color(0xFFC62828),
          title: name,
          subtitle: scientific,
          lines: [
            if (r.pestConfidence != null) S.howSure(r.pestConfidence!),
            r.severity == 'Unclear' || r.severity == null
                ? S.severityNotMeasured
                : S.severityLine(r.severity!, r.damagePercent),
          ],
        );
      case 'not_corn':
        return _Message(icon: Icons.help_outline, color: Colors.grey.shade700, title: S.notCornTitle, lines: const [S.notCornBody]);
      default:
        return _Message(icon: Icons.blur_on, color: Colors.grey.shade700, title: S.unclearTitle, lines: const [S.unclearBody]);
    }
  }
}

class _Message extends StatelessWidget {
  const _Message({required this.icon, required this.color, required this.title, this.subtitle, this.lines = const []});
  final IconData icon;
  final Color color;
  final String title;
  final String? subtitle;
  final List<String> lines;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          children: [
            Icon(icon, size: 56, color: color),
            const SizedBox(height: 8),
            Text(title, textAlign: TextAlign.center, style: theme.textTheme.headlineSmall?.copyWith(color: color)),
            if (subtitle != null && subtitle!.isNotEmpty)
              Text(subtitle!, textAlign: TextAlign.center, style: const TextStyle(fontStyle: FontStyle.italic, fontSize: 16)),
            for (final line in lines) ...[
              const SizedBox(height: 10),
              Text(line, textAlign: TextAlign.center, style: theme.textTheme.bodyLarge),
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
