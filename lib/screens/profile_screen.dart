import 'package:flutter/material.dart';

import '../services/auth_service.dart';
import '../strings.dart';

/// Asked once, after the first sign-in with an email code: name and barangay.
class CompleteProfileScreen extends StatefulWidget {
  const CompleteProfileScreen({super.key, required this.auth, required this.onSaved, this.name});
  final AuthService auth;
  final VoidCallback onSaved;
  final String? name;

  @override
  State<CompleteProfileScreen> createState() => _CompleteProfileScreenState();
}

class _CompleteProfileScreenState extends State<CompleteProfileScreen> {
  final _form = GlobalKey<FormState>();
  late final _name = TextEditingController(text: widget.name ?? '');
  late Future<List<Barangay>> _barangays = widget.auth.barangays();
  int? _barangayId;
  bool _busy = false;
  String? _error;

  @override
  void dispose() {
    _name.dispose();
    super.dispose();
  }

  Future<void> _save() async {
    if (!_form.currentState!.validate()) return;
    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      await widget.auth.updateProfile(fullName: _name.text, barangayId: _barangayId!);
      widget.onSaved();
    } on AuthFailure catch (e) {
      if (mounted) setState(() => _error = e.message);
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(
        title: const Text(S.aboutYou),
        actions: [IconButton(tooltip: S.signOut, icon: const Icon(Icons.logout), onPressed: widget.auth.signOut)],
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(24),
          child: Form(
            key: _form,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Text(S.aboutYouIntro, style: theme.textTheme.bodyLarge),
                const SizedBox(height: 24),
                TextFormField(
                  key: const Key('fullName'),
                  controller: _name,
                  textCapitalization: TextCapitalization.words,
                  decoration: const InputDecoration(labelText: S.fullName, prefixIcon: Icon(Icons.person_outline)),
                  validator: (v) => (v == null || v.trim().isEmpty) ? S.enterName : null,
                ),
                const SizedBox(height: 16),
                FutureBuilder<List<Barangay>>(
                  future: _barangays,
                  builder: (context, snap) {
                    if (snap.hasError) {
                      return Row(children: [
                        const Expanded(child: Text(S.barangaysFailed)),
                        TextButton(
                          onPressed: () => setState(() => _barangays = widget.auth.barangays()),
                          child: const Text(S.tryAgain),
                        ),
                      ]);
                    }
                    final items = snap.data;
                    return DropdownButtonFormField<int>(
                      key: const Key('barangay'),
                      initialValue: _barangayId,
                      isExpanded: true,
                      decoration: InputDecoration(
                        labelText: S.barangay,
                        prefixIcon: const Icon(Icons.place_outlined),
                        helperText: items == null ? S.loadingBarangays : null,
                      ),
                      items: [for (final b in items ?? const <Barangay>[]) DropdownMenuItem(value: b.id, child: Text(b.name))],
                      onChanged: items == null ? null : (v) => setState(() => _barangayId = v),
                      validator: (v) => v == null ? S.chooseBarangay : null,
                    );
                  },
                ),
                if (_error != null) ...[
                  const SizedBox(height: 16),
                  Text(_error!, style: TextStyle(color: theme.colorScheme.error, fontSize: 16)),
                ],
                const SizedBox(height: 24),
                FilledButton(
                  onPressed: _busy ? null : _save,
                  child: _busy
                      ? const SizedBox(width: 24, height: 24, child: CircularProgressIndicator(strokeWidth: 3))
                      : const Text(S.save),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
