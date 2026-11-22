import 'package:flutter/material.dart';

import '../services/app_services.dart';
import '../services/auth_service.dart';
import '../services/history_store.dart';
import '../strings.dart';
import 'profile_screen.dart';

/// The Settings tab: the user's info (and changing it), uploads, app info, sign out.
class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key, required this.services, this.onProfileChanged, this.modelVersion});
  final AppServices services;
  final VoidCallback? onProfileChanged;
  final Future<String?>? modelVersion;

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  AuthService get _auth => widget.services.auth;
  HistoryStore get _history => widget.services.history;
  late Future<Profile> _profile = _auth.profile();
  late Future<List<LocalScan>> _scans = _history.list();
  bool _uploading = false;

  @override
  void initState() {
    super.initState();
    _history.changes.addListener(_reloadScans);
  }

  @override
  void dispose() {
    _history.changes.removeListener(_reloadScans);
    super.dispose();
  }

  void _reloadScans() {
    if (!mounted) return;
    setState(() {
      _scans = _history.list();
    });
  }

  void _toast(String text) => ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(text)));

  Future<void> _editProfile(Profile p) async {
    await Navigator.of(context).push(MaterialPageRoute(
      builder: (context) => CompleteProfileScreen(
        auth: _auth,
        firstTime: false,
        name: p.fullName,
        barangayId: p.barangayId,
        onSaved: () => Navigator.of(context).pop(),
      ),
    ));
    if (!mounted) return;
    setState(() {
      _profile = _auth.profile();
    });
    widget.onProfileChanged?.call();
  }

  Future<void> _changePassword() async {
    final changed = await showDialog<bool>(context: context, builder: (_) => _PasswordDialog(auth: _auth));
    if (changed == true && mounted) _toast(S.passwordChanged);
  }

  Future<void> _uploadNow() async {
    setState(() => _uploading = true);
    final report = await _history.sync();
    if (!mounted) return;
    setState(() => _uploading = false);
    _toast(report.waiting == 0 ? S.allUploaded : S.waitingCount(report.waiting));
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
    if (yes != true) return;
    try {
      await _auth.signOut();
    } on AuthFailure catch (e) {
      if (mounted) _toast(e.message);
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(title: const Text(S.settings)),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(16, 8, 16, 24),
        children: [
          FutureBuilder<Profile>(
            future: _profile,
            builder: (context, snap) {
              final p = snap.data ?? const Profile();
              final name = (p.fullName ?? '').trim();
              return Card(
                child: Column(children: [
                  ListTile(
                    contentPadding: const EdgeInsets.fromLTRB(16, 12, 16, 4),
                    leading: CircleAvatar(
                      radius: 26,
                      backgroundColor: theme.colorScheme.primary,
                      child: Text(name.isEmpty ? '?' : name[0].toUpperCase(),
                          style: const TextStyle(color: Colors.white, fontSize: 22)),
                    ),
                    title: Text(name.isEmpty ? S.notSet : name, style: theme.textTheme.titleMedium),
                    subtitle: Text(p.email ?? ''),
                  ),
                  ListTile(
                    leading: const Icon(Icons.place_outlined),
                    title: const Text(S.barangay),
                    subtitle: Text(p.barangay ?? S.notSet),
                  ),
                  const Divider(height: 1),
                  ListTile(
                    leading: const Icon(Icons.edit_outlined),
                    title: const Text(S.editProfile),
                    trailing: const Icon(Icons.chevron_right),
                    enabled: snap.hasData,
                    onTap: () => _editProfile(p),
                  ),
                  ListTile(
                    leading: const Icon(Icons.lock_outline),
                    title: const Text(S.changePassword),
                    trailing: const Icon(Icons.chevron_right),
                    onTap: _changePassword,
                  ),
                ]),
              );
            },
          ),
          const SizedBox(height: 8),
          FutureBuilder<List<LocalScan>>(
            future: _scans,
            builder: (context, snap) {
              final waiting = (snap.data ?? const <LocalScan>[]).where((s) => !s.synced).length;
              return Card(
                child: ListTile(
                  leading: Icon(waiting == 0 ? Icons.cloud_done : Icons.cloud_upload_outlined,
                      color: waiting == 0 ? theme.colorScheme.primary : Colors.grey.shade700),
                  title: Text(waiting == 0 ? S.allUploaded : S.waitingCount(waiting)),
                  trailing: waiting == 0
                      ? null
                      : (_uploading
                          ? const SizedBox(width: 24, height: 24, child: CircularProgressIndicator(strokeWidth: 3))
                          : TextButton(onPressed: _uploadNow, child: const Text(S.uploadNow))),
                ),
              );
            },
          ),
          const SizedBox(height: 8),
          Card(
            child: Column(children: [
              const ListTile(leading: Icon(Icons.info_outline), title: Text(S.aboutApp), subtitle: Text(S.worksOffline)),
              if (widget.modelVersion != null)
                FutureBuilder<String?>(
                  future: widget.modelVersion,
                  builder: (context, snap) => snap.data == null
                      ? const SizedBox.shrink()
                      : ListTile(leading: const Icon(Icons.memory), title: Text(S.modelVersion(snap.data!))),
                ),
            ]),
          ),
          const SizedBox(height: 16),
          OutlinedButton.icon(
            style: OutlinedButton.styleFrom(foregroundColor: theme.colorScheme.error),
            onPressed: _signOut,
            icon: const Icon(Icons.logout),
            label: const Text(S.signOut),
          ),
        ],
      ),
    );
  }
}

class _PasswordDialog extends StatefulWidget {
  const _PasswordDialog({required this.auth});
  final AuthService auth;

  @override
  State<_PasswordDialog> createState() => _PasswordDialogState();
}

class _PasswordDialogState extends State<_PasswordDialog> {
  final _form = GlobalKey<FormState>();
  final _pw = TextEditingController(), _repeat = TextEditingController();
  bool _show = false, _busy = false;
  String? _error;

  @override
  void dispose() {
    _pw.dispose();
    _repeat.dispose();
    super.dispose();
  }

  Future<void> _save() async {
    if (!_form.currentState!.validate()) return;
    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      await widget.auth.changePassword(_pw.text);
      if (mounted) Navigator.of(context).pop(true);
    } on AuthFailure catch (e) {
      if (mounted) setState(() => _error = e.message);
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: const Text(S.changePassword),
      content: Form(
        key: _form,
        child: Column(mainAxisSize: MainAxisSize.min, children: [
          TextFormField(
            key: const Key('newPassword'),
            controller: _pw,
            obscureText: !_show,
            decoration: InputDecoration(
              labelText: S.newPassword,
              suffixIcon: IconButton(
                tooltip: _show ? S.hidePassword : S.showPassword,
                icon: Icon(_show ? Icons.visibility_off : Icons.visibility),
                onPressed: () => setState(() => _show = !_show),
              ),
            ),
            validator: (v) => (v == null || v.length < 6) ? S.passwordTooShort : null,
          ),
          const SizedBox(height: 12),
          TextFormField(
            key: const Key('repeatPassword'),
            controller: _repeat,
            obscureText: !_show,
            decoration: const InputDecoration(labelText: S.confirmPassword),
            validator: (v) => v != _pw.text ? S.passwordsDiffer : null,
          ),
          if (_error != null) ...[
            const SizedBox(height: 12),
            Text(_error!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
          ],
        ]),
      ),
      actions: [
        TextButton(onPressed: _busy ? null : () => Navigator.of(context).pop(false), child: const Text(S.cancel)),
        FilledButton(onPressed: _busy ? null : _save, child: const Text(S.save)),
      ],
    );
  }
}
