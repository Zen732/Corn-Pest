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
  static const showPassword = 'Show password';
  static const hidePassword = 'Hide password';
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

  // Scan: photo tips and taking the photo
  static const photoTips = 'Photo tips';
  static const tips = [
    'Get close: fill the photo with one leaf, the whorl or the stalk.',
    'If you see an insect, put it in the middle of the photo.',
    'Use daylight or shade. Avoid very dark or very bright photos.',
    'Keep the phone still so the photo is sharp.',
  ];
  static const dontShowTips = 'Don\'t show these tips again';
  static const showTips = 'Show photo tips';
  static const takePhoto = 'Take a photo';
  static const chooseFromGallery = 'Choose from gallery';

  // Checking the photo
  static const checking = 'Checking your photo...';
  static const backendMissing = 'The app has no checking server address (BACKEND_URL in env.json).';
  static const serverUnreachable =
      'Can\'t reach the checking server. Make sure it is running and your phone is on the same Wi-Fi.';
  static const serverTooSlow = 'The checking server took too long. Try again.';
  static const checkFailed = 'The photo couldn\'t be checked. Try again.';
  static const photoUnreadable = 'This photo couldn\'t be opened. Try another photo.';
  static const modelsMissing = 'The checking models are missing from this copy of the app.';

  // Result
  static const healthyTitle = 'Healthy';
  static const healthyBody = 'No insect or damage found on this plant.';
  static const damageTitle = 'Damage found';
  static const noInsectSeen = 'No insect seen in this photo.';
  static const notCornTitle = 'This doesn\'t look like a corn plant';
  static const notCornBody = 'Take a photo of a corn leaf, whorl or stalk.';
  static const unclearTitle = 'The plant isn\'t clear in this photo';
  static const unclearBody = 'Get closer and fill the photo with one leaf.';
  static String howSure(double c) => 'How sure: ${(c * 100).round()}%';
  static String severityLine(String level, double? percent) =>
      percent == null ? 'Severity: $level' : 'Severity: $level (${percent.round()}% of the plant damaged)';
  static const severityNotMeasured = 'Severity: not measured. Take a photo of the damaged leaf to measure it.';
  static const pests = {
    'fall_armyworm': ('Fall armyworm', 'Spodoptera frugiperda'),
    'common_cutworm': ('Common cutworm', 'Spodoptera litura'),
    'armyworm': ('Armyworm', 'Mythimna separata'),
    'corn_planthopper': ('Corn planthopper', 'Peregrinus maidis'),
    'corn_aphid': ('Corn aphid', 'Rhopalosiphum maidis'),
  };
  static const saving = 'Saving to History...';
  static const saved = 'Saved to History';
  static const saveFailed = 'Couldn\'t save this result on the phone. Is the storage full?';

  // History
  static const historyEmpty = 'No scans yet. Results you check are saved here.';
  static const uploaded = 'Uploaded';
  static const waitingToUpload = 'Waiting to upload';
  static const waitingNote = 'Saved on this phone. It uploads by itself when you have internet.';
  static String waitingCount(int n) => n == 1 ? '1 scan is waiting to upload.' : '$n scans are waiting to upload.';
  static const photoNotAvailable = 'Photo not on this phone yet. Connect to the internet to see it.';
  static const deleteScan = 'Delete this scan';
  static const deleteQuestion = 'Delete this scan from History?';
  static const delete = 'Delete';
  static const scanDetails = 'Scan details';
  static String scannedOn(DateTime t) {
    final l = t.toLocal();
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    final h = l.hour % 12 == 0 ? 12 : l.hour % 12;
    return '${months[l.month - 1]} ${l.day}, ${l.year}  $h:${l.minute.toString().padLeft(2, '0')} ${l.hour < 12 ? 'AM' : 'PM'}';
  }

  /// One-line summary of a result, for the History list.
  static String summary(String outcome, String? pest, String? severity) {
    switch (outcome) {
      case 'pest':
        final name = pests[pest]?.$1 ?? (pest ?? 'Pest');
        return severity == null || severity == 'Unclear' ? name : '$name - $severity';
      case 'severity_only':
        return 'Damage - ${severity ?? ''}';
      case 'healthy':
        return healthyTitle;
      default:
        return outcome;
    }
  }
  static const retakePhoto = 'Retake photo';
  static const scanAnother = 'Scan another plant';
  static const done = 'Done';
  static const back = 'Back';

  // Settings missing (env.json not passed when building)
  static const setupMissingTitle = 'App settings missing';
  static const setupMissingBody =
      'This copy of the app was built without its settings. Build it with:\n\n'
      'flutter run --dart-define-from-file=env.json';
}
