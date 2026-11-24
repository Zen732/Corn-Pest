// Does the phone's offline code (lib/offline/) give the same answers as Python?
// Needs the reference files from ml/src/parity_reference.py and the models from
// ml/src/export_models.py, which stay on the training PC; elsewhere these tests are skipped.
//     flutter test test/offline_parity_test.dart
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:pest_assessment/offline/decode.dart';
import 'package:pest_assessment/offline/features.dart';
import 'package:pest_assessment/offline/flow.dart';
import 'package:pest_assessment/offline/image_ops.dart';
import 'package:pest_assessment/offline/network.dart';
import 'package:pest_assessment/offline/severity.dart';
import 'package:pest_assessment/offline/svm_model.dart';
import 'package:pest_assessment/services/scan_service.dart';

final parity = Directory('build/parity');
final modelsDir = Directory('assets/models');
const groups = {'color': (0, 44), 'glcm': (44, 54), 'lbp': (54, 82), 'hog': (82, 406)};

SvmModel loadModel(String name) => SvmModel.fromBytes(
    File('${modelsDir.path}/$name.json').readAsStringSync(),
    ByteData.sublistView(File('${modelsDir.path}/$name.bin').readAsBytesSync()));

Network? loadNetwork() => File('${modelsDir.path}/network.bin').existsSync()
    ? Network.fromBytes(File('${modelsDir.path}/network.json').readAsStringSync(),
        ByteData.sublistView(File('${modelsDir.path}/network.bin').readAsBytesSync()))
    : null;

Float32List floats(Object? list) =>
    Float32List.fromList((list as List).map((v) => (v as num).toDouble()).toList());

int argmax(List<double> p) {
  var b = 0;
  for (var i = 1; i < p.length; i++) {
    if (p[i] > p[b]) b = i;
  }
  return b;
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  final skip = !File('${parity.path}/e2e.json').existsSync() || !File('${modelsDir.path}/pest.bin').existsSync()
      ? 'reference files are only on the training PC'
      : null;

  test('same pixels in: Dart features match Python', () {
    final refs = (jsonDecode(File('${parity.path}/raw.json').readAsStringSync()) as List).cast<Map<String, dynamic>>();
    final worst = {for (final g in groups.keys) g: 0.0};
    for (final r in refs) {
      final bytes = File('${parity.path}/raw/${r['file']}').readAsBytesSync();
      final f = extractFeatures(Img(r['width'] as int, r['height'] as int, 3, bytes));
      final ref = (r['features'] as List).cast<num>();
      for (final e in groups.entries) {
        for (var i = e.value.$1; i < e.value.$2; i++) {
          final d = (f[i] - ref[i]).abs();
          if (d > worst[e.key]!) worst[e.key] = d;
        }
      }
    }
    // ignore: avoid_print
    print('largest feature difference over ${refs.length} photos: $worst');
    expect(worst['glcm'], lessThan(1e-6));
    expect(worst['lbp'], lessThan(1e-6));
    expect(worst['hog'], lessThan(1e-5));
    expect(worst['color'], lessThan(1e-3)); // numpy sums float32 in a different order
  }, skip: skip);

  test('real path: Flutter decoder + Dart features + Dart models vs Python', () async {
    final refs = (jsonDecode(File('${parity.path}/e2e.json').readAsStringSync()) as List).cast<Map<String, dynamic>>();
    final models = {for (final k in ['corn_check', 'detector', 'health', 'pest']) k: loadModel(k)};
    final network = loadNetwork();
    var maxCnn = 0.0;
    final same = {for (final k in models.keys) k: 0}, sameFromPyFeatures = {for (final k in models.keys) k: 0};
    var maxFeat = 0.0, maxProbFromPy = 0.0, decodeSamePixels = 0;
    final byGroup = <String, List<int>>{};
    for (final r in refs) {
      final img = await decodePhoto(File(r['path'] as String).readAsBytesSync());
      final f = extractFeatures(img);
      final py = floats(r['features']);
      // V2: Python's numbers per kind, and the phone's (network run on the Flutter-decoded photo).
      final pyNumbers = PhotoNumbers(img, network, {
        'current': py,
        if (r['cnn'] != null) 'cnn': floats(r['cnn']),
        if (r['cnn_mirror'] != null) 'cnn_mirror': floats(r['cnn_mirror']),
      });
      final ourNumbers = PhotoNumbers(img, network, {'current': f});
      if (r['cnn'] != null && network != null) {
        final a = ourNumbers.of(models['corn_check']!), b = floats(r['cnn']);
        if (models['corn_check']!.features == 'cnn') {
          for (var i = 0; i < b.length; i++) {
            final dd = (a[i] - b[i]).abs();
            if (dd > maxCnn) maxCnn = dd;
          }
        }
      }
      var d = 0.0;
      for (var i = 0; i < featureLength; i++) {
        d = (f[i] - py[i]).abs() > d ? (f[i] - py[i]).abs() : d;
      }
      if (d > maxFeat) maxFeat = d;
      if (d < 1e-3) decodeSamePixels++;
      var allSame = true;
      for (final e in models.entries) {
        final pyProbs = ((r['probs'] as Map)[e.key] as List).map((v) => (v as num).toDouble()).toList();
        final fromPy = e.value.probabilities(pyNumbers.of(e.value)); // Dart model on Python's numbers
        for (var j = 0; j < pyProbs.length; j++) {
          final diff = (fromPy[j] - pyProbs[j]).abs();
          if (diff > maxProbFromPy) maxProbFromPy = diff;
        }
        if (argmax(fromPy) == argmax(pyProbs)) sameFromPyFeatures[e.key] = sameFromPyFeatures[e.key]! + 1;
        final ours = e.value.probabilities(ourNumbers.of(e.value)); // Dart model on Dart's numbers (the phone)
        if (argmax(ours) == argmax(pyProbs)) {
          same[e.key] = same[e.key]! + 1;
        } else {
          allSame = false;
        }
      }
      final g = byGroup.putIfAbsent(r['group'] as String, () => [0, 0]);
      g[0]++;
      if (allSame) g[1]++;
    }
    final n = refs.length;
    String pct(int k) => '${(100 * k / n).toStringAsFixed(1)}%';
    // ignore: avoid_print
    print('$n photos. Dart models on Python features: max probability difference $maxProbFromPy, '
        'same answer ${sameFromPyFeatures.map((k, v) => MapEntry(k, pct(v)))}\n'
        'Full phone path (Flutter decoder + Dart features): same answer ${same.map((k, v) => MapEntry(k, pct(v)))}; '
        'features within 0.001 of Python on ${pct(decodeSamePixels)} of photos (largest difference $maxFeat); '
        'network numbers largest difference $maxCnn\n'
        'all four models agree with Python, by kind: ${byGroup.map((k, v) => MapEntry(k, '${v[1]}/${v[0]}'))}');
    for (final k in models.keys) {
      expect(sameFromPyFeatures[k], n, reason: '$k: Dart model math must match Python exactly');
    }
    expect(maxProbFromPy, lessThan(1e-6));
  }, skip: skip, timeout: const Timeout(Duration(minutes: 30)));

  test('the app\'s offline checker (models from the app bundle) gives Python\'s answers', () async {
    final refs = (jsonDecode(File('${parity.path}/e2e.json').readAsStringSync()) as List).cast<Map<String, dynamic>>();
    final api = OfflineScanApi();
    for (final r in refs.take(20)) {
      final result = await api.check(File(r['path'] as String).readAsBytesSync());
      final py = r['flow'] as Map<String, dynamic>;
      expect(result.outcome, py['outcome'], reason: r['path'] as String);
      expect(result.pest, py['pest'], reason: r['path'] as String);
      expect(result.modelVersion, isNotEmpty);
    }
  }, skip: skip, timeout: const Timeout(Duration(minutes: 10)));

  test('whole app on the phone (incl. severity) vs Python', () async {
    final refs = (jsonDecode(File('${parity.path}/e2e.json').readAsStringSync()) as List).cast<Map<String, dynamic>>();
    final models = OfflineModels(
        cornCheck: loadModel('corn_check'), detector: loadModel('detector'), pest: loadModel('pest'),
        health: loadModel('health'), version: 'test', network: loadNetwork());
    var sameOutcome = 0, samePest = 0, sameLevel = 0, levelCompared = 0, samePlantShare = 0;
    final damageDiffs = <double>[];
    final levelMisses = <String>[];
    final timer = Stopwatch()..start();
    for (final r in refs) {
      final img = await decodePhoto(File(r['path'] as String).readAsBytesSync());
      final py = r['flow'] as Map<String, dynamic>, pySev = r['severity'] as Map<String, dynamic>;
      final ours = assessPhoto(img, models);
      final sev = assessSeverity(img);
      if (ours['outcome'] == py['outcome']) sameOutcome++;
      if (ours['pest'] == py['pest']) samePest++;
      if ((sev.plantShare - ((pySev['plant_share'] as num?)?.toDouble() ?? -1)).abs() < 1e-9 ||
          (sev.damagePercent == null && pySev['damage_percent'] == null)) {
        samePlantShare++;
      } else {
        // ignore: avoid_print
        print('plant area differs: ${r['path']} ${img.width}x${img.height} phone ${sev.plantShare} '
            '(damage ${sev.damagePercent}) python ${pySev['plant_share']} (damage ${pySev['damage_percent']})');
      }
      if (py['severity'] != null && ours['outcome'] == py['outcome']) {
        levelCompared++;
        if (ours['severity'] == py['severity']) {
          sameLevel++;
        } else if (levelMisses.length < 8) {
          levelMisses.add('${py['severity']}->${ours['severity']} (${py['damage_percent']} vs ${ours['damage_percent']})');
        }
      }
      if (sev.damagePercent != null && pySev['damage_percent'] != null) {
        damageDiffs.add((sev.damagePercent! - (pySev['damage_percent'] as num).toDouble()).abs());
      }
    }
    final n = refs.length;
    damageDiffs.sort();
    double q(double p) => damageDiffs[((damageDiffs.length - 1) * p).round()];
    String pct(int k, int of) => '${(100 * k / of).toStringAsFixed(1)}% ($k/$of)';
    // ignore: avoid_print
    print('$n photos, ${(timer.elapsedMilliseconds / n).toStringAsFixed(0)} ms per photo here.\n'
        'same outcome ${pct(sameOutcome, n)}, same pest ${pct(samePest, n)}, '
        'same plant area (no K-Means involved) ${pct(samePlantShare, n)}\n'
        'same severity grade ${pct(sameLevel, levelCompared)}; damage % difference: median ${q(0.5)}, '
        '90% under ${q(0.9)}, largest ${damageDiffs.last}\n'
        'grade differences (Python -> phone): $levelMisses');
    expect(samePlantShare, n, reason: 'background and holes must match Python exactly');
  }, skip: skip, timeout: const Timeout(Duration(minutes: 60)));
}
