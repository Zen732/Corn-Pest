import 'package:flutter/material.dart';

import '../services/auth_service.dart';
import '../strings.dart';
import 'sign_in_screen.dart' show validateEmail;

/// The eye icon that shows or hides a password.
class _EyeButton extends StatelessWidget {
  const _EyeButton({super.key, required this.visible, required this.onPressed});
  final bool visible;
  final VoidCallback onPressed;

  @override
  Widget build(BuildContext context) => IconButton(
        tooltip: visible ? S.hidePassword : S.showPassword,
        icon: Icon(visible ? Icons.visibility_off : Icons.visibility),
        onPressed: onPressed,
      );
}

class SignUpScreen extends StatefulWidget {
  const SignUpScreen({super.key, required this.auth});
  final AuthService auth;

  @override
  State<SignUpScreen> createState() => _SignUpScreenState();
}

class _SignUpScreenState extends State<SignUpScreen> {
  final _form = GlobalKey<FormState>();
  final _name = TextEditingController();
  final _email = TextEditingController();
  final _password = TextEditingController();
  final _repeat = TextEditingController();
  late Future<List<Barangay>> _barangays;
  int? _barangayId;
  bool _showPassword = false;
  bool _showRepeat = false;
  bool _busy = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    _barangays = widget.auth.barangays();
  }

  @override
  void dispose() {
    for (final c in [_name, _email, _password, _repeat]) {
      c.dispose();
    }
    super.dispose();
  }

  Future<void> _create() async {
    if (!_form.currentState!.validate()) return;
    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      final signedIn = await widget.auth.signUp(
        fullName: _name.text,
        barangayId: _barangayId!,
        email: _email.text,
        password: _password.text,
      );
      if (!mounted) return;
      setState(() => _busy = false);
      if (!signedIn) {
        await showDialog<void>(
          context: context,
          builder: (context) => AlertDialog(
            title: const Text(S.checkEmailTitle),
            content: const Text(S.checkEmailBody),
            actions: [TextButton(onPressed: () => Navigator.of(context).pop(), child: const Text(S.ok))],
          ),
        );
      }
      // Back to the start: Home if signed in, otherwise Sign in.
      if (mounted) Navigator.of(context).popUntil((route) => route.isFirst);
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
      appBar: AppBar(title: const Text(S.createAccount)),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(24),
          child: Form(
            key: _form,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
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
                      items: [
                        for (final b in items ?? const <Barangay>[])
                          DropdownMenuItem(value: b.id, child: Text(b.name)),
                      ],
                      onChanged: items == null ? null : (v) => setState(() => _barangayId = v),
                      validator: (v) => v == null ? S.chooseBarangay : null,
                    );
                  },
                ),
                const SizedBox(height: 16),
                TextFormField(
                  key: const Key('email'),
                  controller: _email,
                  keyboardType: TextInputType.emailAddress,
                  decoration: const InputDecoration(labelText: S.email, prefixIcon: Icon(Icons.email_outlined)),
                  validator: validateEmail,
                ),
                const SizedBox(height: 16),
                TextFormField(
                  key: const Key('password'),
                  controller: _password,
                  obscureText: !_showPassword,
                  decoration: InputDecoration(
                    labelText: S.password,
                    prefixIcon: const Icon(Icons.lock_outline),
                    suffixIcon: _EyeButton(
                        key: const Key('showPassword'),
                        visible: _showPassword,
                        onPressed: () => setState(() => _showPassword = !_showPassword)),
                  ),
                  validator: (v) => (v == null || v.length < 6) ? S.passwordTooShort : null,
                ),
                const SizedBox(height: 16),
                TextFormField(
                  key: const Key('repeat'),
                  controller: _repeat,
                  obscureText: !_showRepeat,
                  decoration: InputDecoration(
                    labelText: S.confirmPassword,
                    prefixIcon: const Icon(Icons.lock_outline),
                    suffixIcon: _EyeButton(
                        key: const Key('showRepeat'),
                        visible: _showRepeat,
                        onPressed: () => setState(() => _showRepeat = !_showRepeat)),
                  ),
                  validator: (v) => v != _password.text ? S.passwordsDiffer : null,
                ),
                if (_error != null) ...[
                  const SizedBox(height: 16),
                  Text(_error!, style: TextStyle(color: theme.colorScheme.error, fontSize: 16)),
                ],
                const SizedBox(height: 24),
                FilledButton(
                  onPressed: _busy ? null : _create,
                  child: _busy
                      ? const SizedBox(width: 24, height: 24, child: CircularProgressIndicator(strokeWidth: 3))
                      : const Text(S.createAccount),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
