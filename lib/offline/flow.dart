// The app's decision for one photo, on the phone: the same rules as ml/src/flow.py
//     corn check -> insect? yes: pest classifier (name only) / no: healthy-vs-damaged -> severity
//
// Version 2: each model says which photo numbers it learned from (SvmModel.features): "cnn" =
// the image network's numbers (network.dart; 1408 for EfficientNet-B2), "current" = the
// hand-made features (features.dart). Each set is computed once per photo, only if a model
// needs it.
import 'dart:typed_data';

import 'features.dart';
import 'image_ops.dart';
import 'network.dart';
import 'severity.dart';
import 'svm_model.dart';

/// The pest classifier must be at least this sure to name a pest (PEST_AT_LEAST in flow.py).
const pestAtLeast = 0.50;

/// The corn check must be at least this sure that the photo is corn (CORN_AT_LEAST in flow.py).
const cornAtLeast = 0.50;

/// The pest classifier's answer for "an insect, but not one of the five pests" (OTHER_INSECT).
const otherInsect = 'other_insect';

class OfflineModels {
  const OfflineModels(
      {required this.cornCheck, required this.detector, required this.pest, required this.health, required this.version,
      this.network});
  final SvmModel cornCheck, detector, pest, health;
  final String version;

  /// The image network; needed when a model uses "cnn" numbers.
  final Network? network;
}

double _r3(double x) => roundLikePython(x, 3);

/// The photo mirrored left-right (cv2.flip(rgb, 1)).
Img mirror(Img src) {
  final out = Img(src.width, src.height, src.channels);
  final c = src.channels, w = src.width;
  for (var y = 0; y < src.height; y++) {
    final row = y * w * c;
    for (var x = 0; x < w; x++) {
      final from = row + x * c, to = row + (w - 1 - x) * c;
      for (var k = 0; k < c; k++) {
        out.data[to + k] = src.data[from + k];
      }
    }
  }
  return out;
}

/// The photo numbers each model learned from, each set computed once (photo_numbers in flow.py).
class PhotoNumbers {
  PhotoNumbers(this.rgb, this.network, [Map<String, Float32List>? given]) : _cache = {...?given};
  final Img rgb;
  final Network? network;
  final Map<String, Float32List> _cache;

  Float32List of(SvmModel model) => _get(model.features);

  Float32List _get(String kind) {
    final have = _cache[kind];
    if (have != null) return have;
    final Float32List v;
    if (kind == 'cnn') {
      if (network == null) throw StateError('the image network is not loaded');
      v = network!.vector(rgb);
    } else if (kind == 'cnn_mirror') {
      // The average of the photo's and its mirror image's numbers (the pest classifier).
      final a = _get('cnn'), b = network!.vector(mirror(rgb));
      v = Float32List(a.length);
      for (var i = 0; i < a.length; i++) {
        v[i] = f32(f32(a[i] + b[i]) / 2); // float32 at each step, like numpy
      }
    } else if (kind == 'current') {
      v = extractFeatures(rgb);
    } else {
      final parts = kind.split('+').map(_get).toList();
      v = Float32List(parts.fold(0, (n, p) => n + p.length));
      var at = 0;
      for (final p in parts) {
        v.setAll(at, p);
        at += p.length;
      }
    }
    return _cache[kind] = v;
  }
}

/// Same keys and values as flow.assess_photo in Python (plus model_version), so the result
/// screen and the database use it exactly like the backend's answer. [given]: photo numbers
/// already computed, e.g. {'cnn': ...}.
Map<String, dynamic> assessPhoto(Img rgb, OfflineModels m, {Map<String, Float32List>? given}) {
  final x = PhotoNumbers(rgb, m.network, given);
  final corn = m.cornCheck.predict(x.of(m.cornCheck));
  final result = <String, dynamic>{
    'outcome': 'not_corn', 'corn_confidence': _r3(corn.confidence), 'insect_visible': null,
    'insect_confidence': null, 'pest': null, 'pest_confidence': null, 'healthy': null,
    'health_confidence': null, 'damage_percent': null, 'severity': null, 'model_version': m.version,
  };
  if ((corn.probabilities['corn'] ?? 0) < cornAtLeast) return result;

  final insect = m.detector.predict(x.of(m.detector));
  result['insect_confidence'] = _r3(insect.confidence);
  if (insect.label == 'yes') {
    final pest = m.pest.predict(x.of(m.pest));
    if (pest.label != 'healthy') {
      // An insect: the pest's name only, no damage grade. Another insect, or not sure enough:
      // "not recognized".
      final sure = pest.label != otherInsect && pest.confidence >= pestAtLeast;
      result.addAll({
        'outcome': sure ? 'pest' : 'unknown_pest',
        'insect_visible': true,
        'pest': sure ? pest.label : null,
        'pest_confidence': _r3(pest.confidence),
      });
      return result;
    }
    // The pest classifier says "healthy": the detector was probably wrong; treat as no insect.
  }

  final hp = m.health.predict(x.of(m.health));
  final healthy = hp.label == 'healthy';
  final sev = assessSeverity(rgb);
  final level = healthy ? 'Healthy' : (sev.level == 'Healthy' ? 'Low' : sev.level);
  result.addAll({
    'outcome': healthy ? 'healthy' : (sev.damagePercent == null ? 'unclear' : 'severity_only'),
    'insect_visible': false,
    'healthy': healthy,
    'health_confidence': _r3(hp.confidence),
    'damage_percent': sev.damagePercent,
    'severity': level,
  });
  return result;
}
