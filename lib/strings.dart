/// Every piece of text the app shows, in one place, so a translation (Cebuano, Filipino)
/// can be added later by copying this file.
class S {
  static const appName = 'Corn Pest Assessment';
  static const appTagline = 'Check corn plants for pests and damage';

  // Sign in
  static const signIn = 'Sign in';
  static const email = 'Email';
  static const password = 'Password';
  static const noAccount = 'No account yet?';
  static const createAccount = 'Create account';
  static const enterEmail = 'Enter your email';
  static const invalidEmail = 'This doesn\'t look like an email address';
  static const enterPassword = 'Enter your password';
  static const wrongLogin = 'Wrong email or password.';
  static const emailNotConfirmed = 'Please confirm your email first: open the link we sent you, then sign in.';
  static const couldNotConnect = 'Can\'t connect. Check your internet and try again.';

  // Create account
  static const fullName = 'Full name';
  static const enterName = 'Enter your name';
  static const barangay = 'Barangay (Cabadbaran City)';
  static const chooseBarangay = 'Choose your barangay';
  static const loadingBarangays = 'Loading barangays...';
  static const barangaysFailed = 'Couldn\'t load the barangays.';
  static const tryAgain = 'Try again';
  static const confirmPassword = 'Repeat password';
  static const passwordTooShort = 'Use at least 6 characters';
  static const passwordsDiffer = 'The passwords are not the same';
  static const emailTaken = 'There is already an account with this email. Sign in instead.';
  static const checkEmailTitle = 'Check your email';
  static const checkEmailBody = 'We sent a confirmation link to your email. Open it, then come back and sign in.';
  static const ok = 'OK';

  // Home
  static String hello(String? name) => (name == null || name.isEmpty) ? 'Hello!' : 'Hello, $name!';
  static String fromBarangay(String name) => 'Barangay $name';
  static const scanPlant = 'Scan a corn plant';
  static const history = 'History';
  static const signOut = 'Sign out';
  static const signOutQuestion = 'Sign out of the app?';
  static const cancel = 'Cancel';
  static const comingSoon = 'Coming in the next update.';

  // Settings missing (env.json not passed when building)
  static const setupMissingTitle = 'App settings missing';
  static const setupMissingBody =
      'This copy of the app was built without its settings. Build it with:\n\n'
      'flutter run --dart-define-from-file=env.json';
}
