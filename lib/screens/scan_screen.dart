import 'package:flutter/material.dart';

import '../services/app_services.dart';
import '../strings.dart';
import 'result_screen.dart';

/// Photo tips, then "Take a photo" or "Choose from gallery".
class ScanScreen extends StatefulWidget {
  const ScanScreen({super.key, required this.services});
  final AppServices services;

  @override
  State<ScanScreen> createState() => _ScanScreenState();
}

class _ScanScreenState extends State<ScanScreen> {
  late bool _showTips = !widget.services.tips.hideTips;
  bool _picking = false;

  static const _tipIcons = [Icons.crop_free, Icons.bug_report_outlined, Icons.wb_sunny_outlined, Icons.back_hand_outlined];

  Future<void> _pick({required bool camera}) async {
    setState(() => _picking = true);
    try {
      final photo = await widget.services.photos.pick(camera: camera);
      if (photo == null || !mounted) return; // cancelled
      await Navigator.of(context).push(
        MaterialPageRoute(builder: (_) => ResultScreen(services: widget.services, photo: photo)),
      );
    } finally {
      if (mounted) setState(() => _picking = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(title: const Text(S.scanPlant)),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(24),
          children: [
            if (_showTips)
              Card(
                child: Padding(
                  padding: const EdgeInsets.fromLTRB(16, 16, 16, 4),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(S.photoTips, style: theme.textTheme.titleLarge),
                      const SizedBox(height: 8),
                      for (var i = 0; i < S.tips.length; i++)
                        ListTile(
                          contentPadding: EdgeInsets.zero,
                          leading: Icon(_tipIcons[i], color: theme.colorScheme.primary),
                          title: Text(S.tips[i], style: theme.textTheme.bodyLarge),
                        ),
                      CheckboxListTile(
                        key: const Key('hideTips'),
                        contentPadding: EdgeInsets.zero,
                        controlAffinity: ListTileControlAffinity.leading,
                        value: widget.services.tips.hideTips,
                        title: const Text(S.dontShowTips),
                        onChanged: (v) async {
                          await widget.services.tips.setHideTips(v ?? false);
                          setState(() {});
                        },
                      ),
                    ],
                  ),
                ),
              )
            else
              Align(
                alignment: Alignment.centerLeft,
                child: TextButton.icon(
                  onPressed: () => setState(() => _showTips = true),
                  icon: const Icon(Icons.lightbulb_outline),
                  label: const Text(S.showTips),
                ),
              ),
            const SizedBox(height: 24),
            SizedBox(
              height: 96,
              child: FilledButton.icon(
                onPressed: _picking ? null : () => _pick(camera: true),
                icon: const Icon(Icons.photo_camera, size: 32),
                label: const Text(S.takePhoto, style: TextStyle(fontSize: 22)),
              ),
            ),
            const SizedBox(height: 16),
            SizedBox(
              height: 64,
              child: OutlinedButton.icon(
                onPressed: _picking ? null : () => _pick(camera: false),
                icon: const Icon(Icons.photo_library_outlined),
                label: const Text(S.chooseFromGallery),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
