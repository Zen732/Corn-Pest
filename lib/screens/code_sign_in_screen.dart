import 'dart:async';

import 'package:flutter/material.dart';

import '../services/auth_service.dart';
import '../strings.dart';
import 'sign_in_screen.dart';

/// Sign in (or sign up) with a 6-digit code sent by email: no password to remember.
/// A new account is made automatically; its name and barangay are asked next (profile_screen).
class CodeSignInScreen extends StatefulWidget {
  const CodeSignInScreen({super.key, required this.auth});
  final AuthService auth;

  @override
  State<CodeSignInScreen> createState() => _CodeSignInScreenState();
}

class _CodeSignInScreenState extends State<CodeSignInScreen> {
  final _form = GlobalKey<FormState>();
  final _email = TextEditingController();
  final _code = TextEditingController();
  String? _sentTo; // the email the code went to; null = still asking for the email
  bool _busy = false;
  String? _error;
  int _wait = 0; // seconds before a new code may be asked for
  Timer? _timer;

  @override
  void dispose() {
    _timer?.cancel();
    _email.dispose();
    _code.dispose();
    super.dispose();
  }

  Future<void> _run(Future<void> Function() action) async {
    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      await action();
    } on AuthFailure catch (e) {
      if (mounted) setState(() => _error = e.message);
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _send() async {
    if (_sentTo == null && !_form.currentState!.validate()) return;
    final email = _sentTo ?? _email.text.trim();
    await _run(() async {
      await widget.auth.sendCode(email);
      if (!mounted) return;
      setState(() {
        _sentTo = email;
        _wait = 60;
      });
      _timer?.cancel();
      _timer = Timer.periodic(const Duration(seconds: 1), (t) {
        if (!mounted || _wait <= 1) t.cancel();
        if (mounted) setState(() => _wait = _wait > 0 ? _wait - 1 : 0);
      });
    });
  }

  Future<void> _verify() async {
    if (!_form.currentState!.validate()) return;
    // The app switches to the next screen by itself once signed in.
    await _run(() async {
      await widget.auth.verifyCode(email: _sentTo!, code: _code.text);
      if (mounted) Navigator.of(context).popUntil((route) => route.isFirst);
    });
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final sent = _sentTo != null;
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
                  if (!sent) ...[
                    Text(S.codeIntro, style: theme.textTheme.bodyLarge),
                    const SizedBox(height: 16),
                    TextFormField(
                      key: const Key('email'),
                      controller: _email,
                      keyboardType: TextInputType.emailAddress,
                      autofillHints: const [AutofillHints.email],
                      decoration: const InputDecoration(labelText: S.email, prefixIcon: Icon(Icons.email_outlined)),
                      validator: validateEmail,
                      onFieldSubmitted: (_) => _send(),
                    ),
                  ] else ...[
                    Text(S.codeSentTo(_sentTo!), style: theme.textTheme.bodyLarge),
                    const SizedBox(height: 16),
                    TextFormField(
                      key: const Key('code'),
                      controller: _code,
                      keyboardType: TextInputType.number,
                      autofillHints: const [AutofillHints.oneTimeCode],
                      maxLength: 10,
                      style: const TextStyle(fontSize: 24, letterSpacing: 6),
                      decoration: const InputDecoration(labelText: S.code, prefixIcon: Icon(Icons.pin_outlined), counterText: ''),
                      validator: (v) => RegExp(r'^\d{6,10}$').hasMatch((v ?? '').trim()) ? null : S.enterCode,
                      onFieldSubmitted: (_) => _verify(),
                    ),
                  ],
                  if (_error != null) ...[
                    const SizedBox(height: 16),
                    Text(_error!, style: TextStyle(color: theme.colorScheme.error, fontSize: 16)),
                  ],
                  const SizedBox(height: 24),
                  FilledButton(
                    onPressed: _busy ? null : (sent ? _verify : _send),
                    child: _busy
                        ? const SizedBox(width: 24, height: 24, child: CircularProgressIndicator(strokeWidth: 3))
                        : Text(sent ? S.signIn : S.sendCode),
                  ),
                  if (sent) ...[
                    const SizedBox(height: 8),
                    TextButton(
                      onPressed: _busy || _wait > 0 ? null : _send,
                      child: Text(_wait > 0 ? S.resendIn(_wait) : S.resendCode),
                    ),
                    TextButton(
                      onPressed: _busy
                          ? null
                          : () => setState(() {
                                _sentTo = null;
                                _code.clear();
                                _error = null;
                              }),
                      child: const Text(S.changeEmail),
                    ),
                  ],
                  const SizedBox(height: 16),
                  TextButton(
                    onPressed: _busy
                        ? null
                        : () => Navigator.of(context).canPop()
                            ? Navigator.of(context).pop()
                            : Navigator.of(context)
                                .push(MaterialPageRoute(builder: (_) => SignInScreen(auth: widget.auth))),
                    child: const Text(S.usePassword),
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
