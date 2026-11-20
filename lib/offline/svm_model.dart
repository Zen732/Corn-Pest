// A trained model exported by ml/src/export_models.py, run on the phone: StandardScaler, RBF
// SVM (libsvm one-vs-one), probabilities (libsvm pairwise coupling, or a sigmoid calibration),
// then the pest sub-class merge and cutoffs from ml/src/predict.py.
import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

class Prediction {
  const Prediction(this.label, this.confidence, this.probabilities);
  final String label;
  final double confidence;
  final Map<String, double> probabilities;
}

class SvmModel {
  SvmModel._({
    required this.classes,
    required this.modelClasses,
    required this.merge,
    required this.cutoffs,
    required this.mean,
    required this.scale,
    required this.gamma,
    required this.nSupport,
    required this.intercept,
    required this.probA,
    required this.probB,
    required this.calibrator,
    required this.sv,
    required this.dualCoef,
    required this.featureLength,
  });

  /// [meta] is the model's .json file, [bin] its .bin file.
  factory SvmModel.fromBytes(String meta, ByteData bin) {
    final m = jsonDecode(meta) as Map<String, dynamic>;
    final dims = m['feature_length'] as int;
    final nSupport = (m['n_support'] as List).cast<int>();
    final nSv = nSupport.fold<int>(0, (a, b) => a + b);
    final k = nSupport.length;
    final sv = Float32List(nSv * dims);
    for (var i = 0; i < sv.length; i++) {
      sv[i] = bin.getFloat32(i * 4, Endian.little);
    }
    final coefOffset = sv.length * 4;
    final dual = Float64List((k - 1) * nSv);
    for (var i = 0; i < dual.length; i++) {
      dual[i] = bin.getFloat64(coefOffset + i * 8, Endian.little);
    }
    List<double> doubles(String key) => (m[key] as List).map((v) => (v as num).toDouble()).toList();
    final cal = m['calibrator'] as Map<String, dynamic>?;
    return SvmModel._(
      classes: (m['classes'] as List).cast<String>(),
      modelClasses: (m['model_classes'] as List).cast<String>(),
      merge: (m['merge'] as Map).cast<String, String>(),
      cutoffs: (m['cutoffs'] as Map).map((k, v) => MapEntry(k as String, (v as num).toDouble())),
      mean: doubles('scaler_mean'),
      scale: doubles('scaler_scale'),
      gamma: (m['gamma'] as num).toDouble(),
      nSupport: nSupport,
      intercept: doubles('intercept'),
      probA: doubles('prob_a'),
      probB: doubles('prob_b'),
      calibrator: cal == null ? null : ((cal['a'] as num).toDouble(), (cal['b'] as num).toDouble()),
      sv: sv,
      dualCoef: dual,
      featureLength: dims,
    );
  }

  final List<String> classes, modelClasses;
  final Map<String, String> merge;
  final Map<String, double> cutoffs;
  final List<double> mean, scale, intercept, probA, probB;
  final double gamma;
  final List<int> nSupport;
  final (double, double)? calibrator;
  final Float32List sv;
  final Float64List dualCoef;
  final int featureLength;

  /// Probabilities in [classes] order (after merging sub-classes).
  List<double> probabilities(Float32List features) {
    final x = Float64List(featureLength);
    final f = Float32List(1);
    for (var d = 0; d < featureLength; d++) {
      f[0] = (features[d] - mean[d]) / scale[d]; // scikit-learn scales float32 features in float32
      x[d] = f[0];
    }
    final dec = _decisions(x);
    final List<double> proba;
    if (calibrator != null) {
      final df = -dec[0]; // scikit-learn's yes/no decision value is -libsvm's
      final p1 = 1.0 / (1.0 + math.exp(calibrator!.$1 * df + calibrator!.$2));
      proba = [1 - p1, p1];
    } else {
      final k = nSupport.length;
      final r = List.generate(k, (_) => List<double>.filled(k, 0));
      var p = 0;
      for (var i = 0; i < k; i++) {
        for (var j = i + 1; j < k; j++) {
          final rij = _sigmoidPredict(dec[p], probA[p], probB[p]).clamp(1e-7, 1 - 1e-7);
          r[i][j] = rij;
          r[j][i] = 1 - rij;
          p++;
        }
      }
      proba = _multiclassProbability(r);
    }
    final out = List<double>.filled(classes.length, 0);
    for (var j = 0; j < modelClasses.length; j++) {
      out[classes.indexOf(merge[modelClasses[j]] ?? modelClasses[j])] += proba[j];
    }
    return out;
  }

  /// The answer and its confidence, like predict_one in ml/src/predict.py (cutoffs applied).
  Prediction predict(Float32List features) {
    final proba = probabilities(features);
    final masked = List<double>.of(proba);
    for (final e in cutoffs.entries) {
      final j = classes.indexOf(e.key);
      if (j >= 0 && masked[j] < e.value) masked[j] = -1;
    }
    var best = 0;
    for (var j = 1; j < masked.length; j++) {
      if (masked[j] > masked[best]) best = j;
    }
    return Prediction(classes[best], proba[best], {for (var j = 0; j < classes.length; j++) classes[j]: proba[j]});
  }

  List<double> _decisions(Float64List x) {
    final nSv = sv.length ~/ featureLength;
    final kernel = Float64List(nSv);
    for (var s = 0; s < nSv; s++) {
      var d2 = 0.0;
      final base = s * featureLength;
      for (var d = 0; d < featureLength; d++) {
        final diff = sv[base + d] - x[d];
        d2 += diff * diff;
      }
      kernel[s] = math.exp(-gamma * d2);
    }
    final k = nSupport.length;
    final start = List<int>.filled(k + 1, 0);
    for (var i = 0; i < k; i++) {
      start[i + 1] = start[i] + nSupport[i];
    }
    final dec = <double>[];
    var p = 0;
    for (var i = 0; i < k; i++) {
      for (var j = i + 1; j < k; j++) {
        var s = 0.0;
        for (var t = start[i]; t < start[i + 1]; t++) {
          s += dualCoef[(j - 1) * nSv + t] * kernel[t];
        }
        for (var t = start[j]; t < start[j + 1]; t++) {
          s += dualCoef[i * nSv + t] * kernel[t];
        }
        dec.add(s + intercept[p++]);
      }
    }
    return dec;
  }

  static double _sigmoidPredict(double dec, double a, double b) {
    final f = dec * a + b;
    return f >= 0 ? math.exp(-f) / (1.0 + math.exp(-f)) : 1.0 / (1.0 + math.exp(f));
  }

  /// libsvm's pairwise coupling: r[i][j] = P(class i | i or j).
  static List<double> _multiclassProbability(List<List<double>> r) {
    final k = r.length;
    final p = List<double>.filled(k, 1.0 / k);
    final q = List.generate(k, (_) => List<double>.filled(k, 0));
    for (var t = 0; t < k; t++) {
      for (var j = 0; j < t; j++) {
        q[t][t] += r[j][t] * r[j][t];
        q[t][j] = q[j][t];
      }
      for (var j = t + 1; j < k; j++) {
        q[t][t] += r[j][t] * r[j][t];
        q[t][j] = -r[j][t] * r[t][j];
      }
    }
    final eps = 0.005 / k;
    final qp = List<double>.filled(k, 0);
    for (var iter = 0; iter < math.max(100, k); iter++) {
      var pqp = 0.0;
      for (var t = 0; t < k; t++) {
        qp[t] = 0;
        for (var j = 0; j < k; j++) {
          qp[t] += q[t][j] * p[j];
        }
        pqp += p[t] * qp[t];
      }
      var maxError = 0.0;
      for (var t = 0; t < k; t++) {
        maxError = math.max(maxError, (qp[t] - pqp).abs());
      }
      if (maxError < eps) break;
      for (var t = 0; t < k; t++) {
        final diff = (-qp[t] + pqp) / q[t][t];
        p[t] += diff;
        pqp = (pqp + diff * (diff * q[t][t] + 2 * qp[t])) / (1 + diff) / (1 + diff);
        for (var j = 0; j < k; j++) {
          qp[j] = (qp[j] + diff * q[t][j]) / (1 + diff);
          p[j] /= 1 + diff;
        }
      }
    }
    return p;
  }
}
