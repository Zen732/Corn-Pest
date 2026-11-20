// The app's decision for one photo, on the phone: the same rules as ml/src/flow.py
//     corn check -> insect detector -> pest classifier or healthy-vs-damaged -> severity
import 'dart:typed_data';

import 'features.dart';
import 'image_ops.dart';
import 'severity.dart';
import 'svm_model.dart';

/// Pest named but less healthy green leaf than this share of the photo: severity not measured.
const noPlantBelow = 0.05;

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
  if (corn.label != 'corn') return result;

  final insect = m.detector.predict(x);
  var insectVisible = insect.label == 'yes';
  if (insectVisible) {
    final pest = m.pest.predict(x);
    if (pest.label == 'healthy') {
      insectVisible = false; // the detector was probably wrong
    } else {
      result['pest'] = pest.label;
      result['pest_confidence'] = _r3(pest.confidence);
    }
  }
  bool? healthy;
  if (!insectVisible) {
    final hp = m.health.predict(x);
    healthy = hp.label == 'healthy';
    result['healthy'] = healthy;
    result['health_confidence'] = _r3(hp.confidence);
  }

  final sev = assessSeverity(rgb);
  var level = sev.level;
  double? damage = sev.damagePercent;
  if (healthy == true) {
    level = 'Healthy';
  } else if (level == 'Healthy') {
    level = 'Low';
  }
  if (insectVisible && sev.greenShare < noPlantBelow) {
    level = 'Unclear';
    damage = null;
  }
  final String outcome;
  if (insectVisible) {
    outcome = 'pest';
  } else if (healthy == true) {
    outcome = 'healthy';
  } else {
    outcome = sev.damagePercent == null ? 'unclear' : 'severity_only';
  }
  result.addAll({
    'outcome': outcome,
    'insect_visible': insectVisible,
    'insect_confidence': _r3(insect.confidence),
    'damage_percent': damage,
    'severity': level,
  });
  return result;
}
