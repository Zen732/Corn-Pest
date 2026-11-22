// The app's decision for one photo, on the phone: the same rules as ml/src/flow.py
//     corn check -> insect? yes: pest classifier (name only) / no: healthy-vs-damaged -> severity
import 'dart:typed_data';

import 'features.dart';
import 'image_ops.dart';
import 'severity.dart';
import 'svm_model.dart';

/// The pest classifier must be at least this sure to name a pest (PEST_AT_LEAST in flow.py).
const pestAtLeast = 0.50;

/// The corn check must be at least this sure that the photo is corn (CORN_AT_LEAST in flow.py).
const cornAtLeast = 0.60;

class OfflineModels {
  const OfflineModels({required this.cornCheck, required this.detector, required this.pest, required this.health, required this.version});
  final SvmModel cornCheck, detector, pest, health;
  final String version;
}

double _r3(double x) => roundLikePython(x, 3);

/// Same keys and values as flow.assess_photo in Python (plus model_version), so the result
/// screen and the database use it exactly like the backend's answer.
Map<String, dynamic> assessPhoto(Img rgb, OfflineModels m, {Float32List? features}) {
  final x = features ?? extractFeatures(rgb);
  final corn = m.cornCheck.predict(x);
  final result = <String, dynamic>{
    'outcome': 'not_corn', 'corn_confidence': _r3(corn.confidence), 'insect_visible': null,
    'insect_confidence': null, 'pest': null, 'pest_confidence': null, 'healthy': null,
    'health_confidence': null, 'damage_percent': null, 'severity': null, 'model_version': m.version,
  };
  if ((corn.probabilities['corn'] ?? 0) < cornAtLeast) return result;

  final insect = m.detector.predict(x);
  result['insect_confidence'] = _r3(insect.confidence);
  if (insect.label == 'yes') {
    final pest = m.pest.predict(x);
    if (pest.label != 'healthy') {
      // An insect: the pest's name only, no damage grade. Not sure enough: "not recognized".
      final sure = pest.confidence >= pestAtLeast;
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

  final hp = m.health.predict(x);
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
