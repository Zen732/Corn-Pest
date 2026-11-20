import 'dart:async';
import 'dart:convert';
import 'dart:isolate';
import 'dart:math';
import 'dart:typed_data';

import 'package:flutter/services.dart' show AssetBundle, rootBundle;
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

import '../config.dart';
import '../offline/decode.dart';
import '../offline/flow.dart';
import '../offline/image_ops.dart';
import '../offline/svm_model.dart';
import '../strings.dart';

/// The backend's answer for one photo (backend/app/routers/predict.py, ml/src/flow.py).
class ScanResult {
  const ScanResult({
    required this.outcome,
    this.pest,
    this.pestConfidence,
    this.severity,
    this.damagePercent,
    this.insectVisible,
    this.insectConfidence,
    this.healthy,
    this.healthConfidence,
    this.cornConfidence = 0,
    this.modelVersion = 'unknown',
  });

  /// healthy, severity_only, pest, not_corn or unclear.
  final String outcome;
  final String? pest;
  final double? pestConfidence;

  /// Healthy, Low, Moderate, High, or Unclear (a pest with no plant in the photo to measure).
  final String? severity;
  final double? damagePercent;
  final bool? insectVisible;
  final double? insectConfidence;
  final bool? healthy;
  final double? healthConfidence;
  final double cornConfidence;
  final String modelVersion;

  factory ScanResult.fromJson(Map<String, dynamic> j) {
    double? num_(String k) => (j[k] as num?)?.toDouble();
    return ScanResult(
      outcome: j['outcome'] as String,
      pest: j['pest'] as String?,
      pestConfidence: num_('pest_confidence'),
      severity: j['severity'] as String?,
      damagePercent: num_('damage_percent'),
      insectVisible: j['insect_visible'] as bool?,
      insectConfidence: num_('insect_confidence'),
      healthy: j['healthy'] as bool?,
      healthConfidence: num_('health_confidence'),
      cornConfidence: num_('corn_confidence') ?? 0,
      modelVersion: (j['model_version'] as String?) ?? 'unknown',
    );
  }

  /// Only real results go to History; "not corn" and "unclear" ask for a new photo instead.
  bool get savable => const {'healthy', 'severity_only', 'pest'}.contains(outcome);

  Map<String, dynamic> get details => {
        'corn_confidence': cornConfidence,
        'insect_visible': insectVisible,
        'insect_confidence': insectConfidence,
        'healthy': healthy,
        'health_confidence': healthConfidence,
      };
}

/// A problem to show the user, already in plain words.
class ScanFailure implements Exception {
  const ScanFailure(this.message);
  final String message;
  @override
  String toString() => message;
}

/// Sends a photo to the backend and returns its answer.
abstract class ScanApi {
  Future<ScanResult> check(Uint8List photo);
}

class BackendScanApi implements ScanApi {
  BackendScanApi({String? baseUrl, http.Client? client})
      : _base = (baseUrl ?? Config.backendUrl).trim().replaceFirst(RegExp(r'/+$'), ''),
        _client = client ?? http.Client();

  final String _base;
  final http.Client _client;

  @override
  Future<ScanResult> check(Uint8List photo) async {
    if (_base.isEmpty) throw const ScanFailure(S.backendMissing);
    final request = http.MultipartRequest('POST', Uri.parse('$_base/predict'))
      ..files.add(http.MultipartFile.fromBytes('photo', photo, filename: 'photo.jpg'));
    final http.Response res;
    try {
      res = await http.Response.fromStream(await _client.send(request).timeout(const Duration(seconds: 60)));
    } on TimeoutException {
      throw const ScanFailure(S.serverTooSlow);
    } catch (_) {
      throw const ScanFailure(S.serverUnreachable);
    }
    if (res.statusCode == 200) {
      return ScanResult.fromJson(jsonDecode(res.body) as Map<String, dynamic>);
    }
    String? detail;
    try {
      detail = (jsonDecode(res.body) as Map<String, dynamic>)['detail'] as String?;
    } catch (_) {}
    throw ScanFailure(detail ?? S.checkFailed);
  }
}

/// Checks the photo on the phone itself (no internet, no server): the same models and decision
/// as the backend, ported to Dart in lib/offline/ and checked against Python on 903 test photos.
class OfflineScanApi implements ScanApi {
  OfflineScanApi({AssetBundle? bundle}) : _bundle = bundle ?? rootBundle;
  final AssetBundle _bundle;
  Future<OfflineModels>? _models;

  /// Loads the models (about 50 MB) once; call early so the first scan doesn't wait.
  Future<OfflineModels> load() => _models ??= _load();

  Future<OfflineModels> _load() async {
    Future<SvmModel> one(String name) async => SvmModel.fromBytes(
        await _bundle.loadString('assets/models/$name.json', cache: false),
        await _bundle.load('assets/models/$name.bin'));
    return OfflineModels(
      cornCheck: await one('corn_check'),
      detector: await one('detector'),
      pest: await one('pest'),
      health: await one('health'),
      version: (await _bundle.loadString('assets/models/model_version.txt', cache: false)).trim(),
    );
  }

  @override
  Future<ScanResult> check(Uint8List photo) async {
    final OfflineModels models;
    try {
      models = await load();
    } catch (_) {
      _models = null;
      throw const ScanFailure(S.modelsMissing);
    }
    final Img img;
    try {
      img = await decodePhoto(photo);
    } catch (_) {
      throw const ScanFailure(S.photoUnreadable);
    }
    try {
      // In a background isolate, so the screen keeps moving while it works.
      return ScanResult.fromJson(await Isolate.run(() => assessPhoto(img, models)));
    } catch (_) {
      throw const ScanFailure(S.checkFailed);
    }
  }
}

/// Saves a result and its photo to the user's History.
abstract class ScanStore {
  Future<void> save(ScanResult result, Uint8List photo);
}

class SupabaseScanStore implements ScanStore {
  SupabaseClient get _db => Supabase.instance.client;
  final _random = Random();

  @override
  Future<void> save(ScanResult result, Uint8List photo) async {
    final uid = _db.auth.currentUser?.id;
    if (uid == null) throw const ScanFailure(S.saveFailed);
    final isPng = photo.length > 4 && photo[0] == 0x89 && photo[1] == 0x50;
    final name = '${DateTime.now().millisecondsSinceEpoch}_${_random.nextInt(1 << 32).toRadixString(16)}';
    final path = '$uid/$name.${isPng ? 'png' : 'jpg'}'; // each user's own folder (database/schema.sql)
    try {
      await _db.storage.from('scan-photos').uploadBinary(path, photo,
          fileOptions: FileOptions(contentType: isPng ? 'image/png' : 'image/jpeg'));
      await _db.from('scans').insert({
        'photo_path': path,
        'outcome': result.outcome,
        'pest_key': result.outcome == 'pest' ? result.pest : null,
        'pest_confidence': result.outcome == 'pest' ? result.pestConfidence : null,
        'damage_percent': result.damagePercent,
        'severity': result.severity,
        'model_version': result.modelVersion,
        'details': result.details,
        // barangay_id is filled in from the user's profile by the database
      });
    } catch (_) {
      throw const ScanFailure(S.saveFailed);
    }
  }
}

/// Takes a photo with the camera or picks one from the gallery.
abstract class PhotoSource {
  /// The photo's bytes, or null if the user cancelled.
  Future<Uint8List?> pick({required bool camera});
}

class ImagePickerSource implements PhotoSource {
  final _picker = ImagePicker();

  @override
  Future<Uint8List?> pick({required bool camera}) async {
    // Shrunk to at most 1600 px: quick to upload on mobile data. The models work on 256-512 px.
    final file = await _picker.pickImage(
      source: camera ? ImageSource.camera : ImageSource.gallery,
      maxWidth: 1600,
      maxHeight: 1600,
      imageQuality: 85,
    );
    return file?.readAsBytes();
  }
}

/// Remembers "Don't show these tips again".
class TipsPrefs {
  TipsPrefs(this._prefs);
  final SharedPreferences _prefs;
  static const _key = 'hide_photo_tips';

  bool get hideTips => _prefs.getBool(_key) ?? false;
  Future<void> setHideTips(bool hide) => _prefs.setBool(_key, hide);
}
