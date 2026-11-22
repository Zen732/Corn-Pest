import 'package:flutter/material.dart';

import '../services/auth_service.dart';
import '../strings.dart';
import 'sign_up_screen.dart';

String? validateEmail(String? v) {
  final s = v?.trim() ?? '';
  if (s.isEmpty) return S.enterEmail;
  if (!RegExp(r'^[^@\s]+@[^@\s]+\.[^@\s]+$').hasMatch(s)) return S.invalidEmail;
  return null;
}

class SignInScreen extends StatefulWidget {
  const SignInScreen({super.key, required this.auth});
  final AuthService auth;

  @override
  State<SignInScreen> createState() => _SignInScreenState();
}

class _SignInScreenState extends State<SignInScreen> {
  final _form = GlobalKey<FormState>();
  final _email = TextEditingController();
  final _password = TextEditingController();
  bool _busy = false;
  bool _showPassword = false;
  String? _error;

  @override
  void dispose() {
    _email.dispose();
    _password.dispose();
    super.dispose();
  }

  Future<void> _signIn() async {
    if (!_form.currentState!.validate()) return;
    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      await widget.auth.signIn(email: _email.text, password: _password.text);
      // The app shows Home by itself; close this screen if it was opened from the code screen.
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
      body: SafeArea(
        child: Center(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(24),
            child: Form(
              key: _form,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Icon(Icons.eco, size: 72, color: theme.colorScheme.primary),
                  const SizedBox(height: 12),
                  Text(S.appName, textAlign: TextAlign.center, style: theme.textTheme.headlineSmall),
                  const SizedBox(height: 4),
                  Text(S.appTagline, textAlign: TextAlign.center, style: theme.textTheme.bodyLarge),
                  const SizedBox(height: 32),
                  TextFormField(
                    key: const Key('email'),
                    controller: _email,
                    keyboardType: TextInputType.emailAddress,
                    autofillHints: const [AutofillHints.email],
                    decoration: const InputDecoration(labelText: S.email, prefixIcon: Icon(Icons.email_outlined)),
                    validator: validateEmail,
                  ),
                  const SizedBox(height: 16),
                  TextFormField(
                    key: const Key('password'),
                    controller: _password,
                    obscureText: !_showPassword,
                    autofillHints: const [AutofillHints.password],
                    decoration: InputDecoration(
                      labelText: S.password,
                      prefixIcon: const Icon(Icons.lock_outline),
                      suffixIcon: IconButton(
                        tooltip: _showPassword ? S.hidePassword : S.showPassword,
                        icon: Icon(_showPassword ? Icons.visibility_off : Icons.visibility),
                        onPressed: () => setState(() => _showPassword = !_showPassword),
                      ),
                    ),
                    validator: (v) => (v == null || v.isEmpty) ? S.enterPassword : null,
                    onFieldSubmitted: (_) => _signIn(),
                  ),
                  if (_error != null) ...[
                    const SizedBox(height: 16),
                    Text(_error!, style: TextStyle(color: theme.colorScheme.error, fontSize: 16)),
                  ],
                  const SizedBox(height: 24),
                  FilledButton(
                    onPressed: _busy ? null : _signIn,
                    child: _busy
                        ? const SizedBox(width: 24, height: 24, child: CircularProgressIndicator(strokeWidth: 3))
                        : const Text(S.signIn),
                  ),
                  const SizedBox(height: 16),
                  Wrap(
                    alignment: WrapAlignment.center,
                    crossAxisAlignment: WrapCrossAlignment.center,
                    children: [
                      const Text(S.noAccount),
                      TextButton(
                        onPressed: _busy
                            ? null
                            : () => Navigator.of(context).push(
                                MaterialPageRoute(builder: (_) => SignUpScreen(auth: widget.auth))),
                        child: const Text(S.createAccount),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
