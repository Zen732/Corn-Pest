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
  static const emailNotConfirmed = 'Your email isn\'t confirmed yet. Get a code by email to confirm it.';
  static const confirmWithCode = 'Send me a code';
  static const couldNotConnect = 'Can\'t connect. Check your internet and try again.';

  // Sign in with an email code
  static const sendCode = 'Send me a code';
  static const codeIntro = 'Enter your email. We\'ll send you a 6-digit code - no password needed.\n\n'
      'New here? Your account is made automatically.';
  static String codeSentTo(String email) => 'We sent a code to $email. Check your inbox (and spam folder).';
  static const code = 'Code from the email';
  static const enterCode = 'Enter the code from the email';
  static const codeWrong = 'That code is wrong or has expired. Check it, or send a new one.';
  static const codeTooSoon = 'Please wait a minute before asking for another code.';
  static const codeSendFailed = 'The code email couldn\'t be sent. Try again later, or use a password.';
  static const resendCode = 'Send a new code';
  static String resendIn(int s) => 'Send a new code in ${s}s';
  static const changeEmail = 'Use a different email';
  static const usePassword = 'Use a password instead';

  static const useCode = 'Sign in or sign up with an email code';
  static const logIn = 'Log in';

  // Bottom bar
  static const home = 'Home';
  static const settings = 'Settings';

  // Home
  static const statScans = 'Scans';
  static const statPests = 'Pests found';
  static const statWaiting = 'To upload';
  static const recentScans = 'Recent scans';
  static const seeAll = 'See all';
  static const noScansYet = 'No scans yet. Tap the camera button below to check a corn plant.';
  static const tipTitle = 'Photo tip';
  static String today() {
    final d = DateTime.now();
    const days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
    const months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September',
      'October', 'November', 'December'];
    return '${days[d.weekday - 1]}, ${months[d.month - 1]} ${d.day}';
  }

  // Settings
  static const yourAccount = 'Your account';
  static const editProfile = 'Edit name and barangay';
  static const changePassword = 'Change password';
  static const newPassword = 'New password';
  static const passwordChanged = 'Password changed.';
  static const profileSaved = 'Saved.';
  static const uploadNow = 'Upload now';
  static const allUploaded = 'All your scans are uploaded.';
  static const aboutApp = 'About this app';
  static String modelVersion(String v) => 'Pest models from $v';
  static const worksOffline = 'Checks photos on the phone, even without internet. No treatment advice is given.';
  static const notSet = 'Not set';

  // First sign-in: name and barangay
  static const aboutYou = 'About you';
  static const aboutYouIntro = 'Your barangay helps show which pests are found where in Cabadbaran.';
  static const save = 'Save';

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
  static const unknownPestTitle = 'Insect not recognized';
  static const unknownPestBody =
      'An insect was found, but the app can\'t tell which pest it is. It may not be one of the 5 corn pests this app knows.';
  static const unknownPestTip = 'Try a closer, sharper photo of the insect.';
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
        return pests[pest]?.$1 ?? (pest ?? 'Pest');
      case 'unknown_pest':
        return unknownPestTitle;
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
